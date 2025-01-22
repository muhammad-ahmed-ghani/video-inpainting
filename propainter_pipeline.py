from inference_propainter import *

global fix_flow_complete
global fix_raft
global model
global device

device = get_device()
print(f'Running on device: {device}')
##############################################
# set up RAFT and flow competition model
##############################################
ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                model_dir='weights', progress=True, file_name=None)
fix_raft = RAFT_bi(ckpt_path, device)

ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                model_dir='weights', progress=True, file_name=None)
fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
for p in fix_flow_complete.parameters():
    p.requires_grad = False
fix_flow_complete.to(device)
fix_flow_complete.eval()


##############################################
# set up ProPainter model
##############################################
ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                model_dir='weights', progress=True, file_name=None)
model = InpaintGenerator(model_path=ckpt_path).to(device)
model.eval()

def process_video(
    video='inputs/object_removal/bmx-trees',
    mask='inputs/object_removal/bmx-trees_mask',
    output='results',
    resize_ratio=1.0,
    height=-1,
    width=-1,
    mask_dilation=4,
    ref_stride=10,
    neighbor_length=10,
    subvideo_length=80,
    raft_iter=20,
    mode='video_inpainting',
    scale_h=1.0,
    scale_w=1.2,
    save_fps=24,
    save_frames=False,
    fp16=False
):
    global fix_flow_complete
    global fix_raft
    global model
    global device
    
    # Use fp16 precision during inference to reduce running memory cost
    use_half = True if fp16 else False 
    if device == torch.device('cpu'):
        use_half = False

    frames, fps, size, video_name = read_frame_from_videos(video)
    if not width == -1 and not height == -1:
        size = (width, height)
    if not resize_ratio == 1.0:
        size = (int(resize_ratio * size[0]), int(resize_ratio * size[1]))

    frames, size, out_size = resize_frames(frames, size)
    
    fps = save_fps if fps is None else fps
    save_root = os.path.join(output, video_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    if mode == 'video_inpainting':
        frames_len = len(frames)
        flow_masks, masks_dilated = read_mask(mask, frames_len, size, 
                                              flow_mask_dilates=mask_dilation,
                                              mask_dilates=mask_dilation)
        w, h = size
    elif mode == 'video_outpainting':
        assert scale_h is not None and scale_w is not None, 'Please provide a outpainting scale (s_h, s_w).'
        frames, flow_masks, masks_dilated, size = extrapolation(frames, (scale_h, scale_w))
        w, h = size
    else:
        raise NotImplementedError
    
    # for saving the masked frames or video
    masked_frame_for_save = []
    for i in range(len(frames)):
        mask_ = np.expand_dims(np.array(masks_dilated[i]),2).repeat(3, axis=2)/255.
        img = np.array(frames[i])
        green = np.zeros([h, w, 3]) 
        green[:,:,1] = 255
        alpha = 0.6
        # alpha = 1.0
        fuse_img = (1-alpha)*img + alpha*green
        fuse_img = mask_ * fuse_img + (1-mask_)*img
        masked_frame_for_save.append(fuse_img.astype(np.uint8))

    frames_inp = [np.array(f).astype(np.uint8) for f in frames]
    frames = to_tensors()(frames).unsqueeze(0) * 2 - 1    
    flow_masks = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
    frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)
    
    ##############################################
    # ProPainter inference
    ##############################################
    video_length = frames.size(1)
    print(f'\nProcessing: {video_name} [{video_length} frames]...')
    with torch.no_grad():
        # ---- compute flow ----
        if frames.size(-1) <= 640: 
            short_clip_len = 12
        elif frames.size(-1) <= 720: 
            short_clip_len = 8
        elif frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2
        
        # use fp32 for RAFT
        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames[:,f:end_f], iters=raft_iter)
                else:
                    flows_f, flows_b = fix_raft(frames[:,f-1:end_f], iters=raft_iter)
                
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
                
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames, iters=raft_iter)
            torch.cuda.empty_cache()


        if use_half:
            frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
            fix_flow_complete = fix_flow_complete.half()
            model = model.half()

        
        # ---- complete flow ----
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + subvideo_length)
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    flow_masks[:, s_f:e_f+1])
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                    pred_flows_bi_sub, 
                    flow_masks[:, s_f:e_f+1])

                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
                
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
            torch.cuda.empty_cache()
            

        # ---- image propagation ----
        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, subvideo_length) # ensure a minimum of 100 frames for image propagation
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(masked_frames[:, s_f:e_f], 
                                                                       pred_flows_bi_sub, 
                                                                       masks_dilated[:, s_f:e_f], 
                                                                       'nearest')
                updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                    prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                torch.cuda.empty_cache()
                
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()
            
    
    ori_frames = frames_inp
    comp_frames = [None] * video_length

    neighbor_stride = neighbor_length // 2
    if video_length > subvideo_length:
        ref_num = subvideo_length // ref_stride
    else:
        ref_num = -1
    
    # ---- feature propagation + transformer ----
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                                min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
        
        with torch.no_grad():
            # 1.0 indicates mask
            l_t = len(neighbor_ids)
            
            # pred_img = selected_imgs # results of image propagation
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
            
            pred_img = pred_img.view(-1, 3, h, w)

            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                0, 2, 3, 1).numpy().astype(np.uint8)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                    + ori_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else: 
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)
        
        torch.cuda.empty_cache()
                
    # save each frame
    if save_frames:
        for idx in range(video_length):
            f = comp_frames[idx]
            f = cv2.resize(f, out_size, interpolation = cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img_save_root = os.path.join(save_root, 'frames', str(idx).zfill(4)+'.png')
            imwrite(f, img_save_root)
                    

    # if mode == 'video_outpainting':
    #     comp_frames = [i[10:-10,10:-10] for i in comp_frames]
    #     masked_frame_for_save = [i[10:-10,10:-10] for i in masked_frame_for_save]
    
    # save videos frame
    masked_frame_for_save = [cv2.resize(f, out_size) for f in masked_frame_for_save]
    comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
    imageio.mimwrite(os.path.join(save_root, 'masked_in.mp4'), masked_frame_for_save, fps=fps, quality=7)
    imageio.mimwrite(os.path.join(save_root, 'inpaint_out.mp4'), comp_frames, fps=fps, quality=7)
    
    print(f'\nAll results are saved in {save_root}')
    
    torch.cuda.empty_cache()

    return os.path.join(save_root, 'inpaint_out.mp4')