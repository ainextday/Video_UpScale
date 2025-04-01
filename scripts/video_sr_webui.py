"""
Modified from [CodeFormer](https://github.com/sczhou/CodeFormer).
Modified from [้hallo2](https://huggingface.co/fudan-generative-ai/hallo2).
When using or redistributing this feature, please comply with the [S-Lab License 1.0](https://github.com/sczhou/CodeFormer?tab=License-1-ov-file).
We kindly request that you respect the terms of this license in any usage or redistribution of this component.
And also thank you ChatGPT 😄
"""

import gradio as gr

import os
import cv2
import glob
import sys

import torch
from torchvision.transforms.functional import normalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray

from basicsr.utils.registry import ARCH_REGISTRY

import shutil
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
torch.cuda.set_per_process_memory_fraction(0.5, device=0)

def set_realesrgan(tile_size=400, upscale=2):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer
    from basicsr.utils.misc import gpu_is_available
    import urllib.request
    import sys
    import os
    import torch

    def download_with_progress(url, filename):
        def show_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            percent = min(percent, 100)
            sys.stdout.write(f"\r📦 Downloading {os.path.basename(filename)}... {percent}%")
            sys.stdout.flush()
            if percent == 100:
                print(" ✅ Done.")
        urllib.request.urlretrieve(url, filename, reporthook=show_progress)

    use_half = False
    if torch.cuda.is_available():
        no_half_gpu_list = ['1650', '1660']
        if not any(gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list):
            use_half = True

    model_dir = os.path.join(os.path.dirname(__file__), "../pretrained_models/realesrgan")
    os.makedirs(model_dir, exist_ok=True)

    if upscale == 2:
        model_name = "RealESRGAN_x2plus.pth"
        download_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    elif upscale == 4:
        model_name = "RealESRGAN_x4plus.pth"
        download_url = "https://huggingface.co/lllyasviel/Annotators/resolve/main/RealESRGAN_x4plus.pth?download=true"
    elif upscale == 8:
        model_name = "RealESRGAN_x8.pth"
        download_url = "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x8.pth?download=true"
    else:
        raise ValueError("Unsupported upscale value. Please choose 2, 4, or 8.")

    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        print(f"🔄 Model {model_name} not found. Downloading...")
        try:
            download_with_progress(download_url, model_path)
        except Exception as e:
            print(f"\n❌ Failed to download {model_name}: {e}")
            raise

    # ✅ ใช้ RealESRGANer โหลด model โดยตรง (ไม่มี load_model argument)
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4 if upscale == 8 else upscale,
    )
    
    upsampler = RealESRGANer(
        scale=upscale,
        model_path=model_path,  # ✅ ใส่ path ให้มันโหลดเอง
        model=model,
        tile=tile_size,
        tile_pad=40,
        pre_pad=0,
        half=use_half
    )


    if not gpu_is_available():
        import warnings
        warnings.warn(
            'Running on CPU now! Make sure your PyTorch version matches your CUDA. '
            'The unoptimized RealESRGAN is slow on CPU.',
            category=RuntimeWarning
        )

    return upsampler




def read_input_images(args):
    img_list = []

    if args.input_path.endswith(('.mp4', '.mov', '.avi', '.mkv')):
        cap = cv2.VideoCapture(args.input_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_list.append(frame)
        cap.release()
    else:
        img_list = sorted(glob.glob(os.path.join(args.input_path, '*')))
        img_list = [cv2.imread(img, cv2.IMREAD_COLOR) for img in img_list]

    return img_list

def process_video_sr(input_path, fidelity_weight, upscale, bg_upsampler, face_upsample, progress=gr.Progress(track_tqdm=True)):
    import sys
    from types import SimpleNamespace

    device = get_device()

    print()
    print("============================================================")
    print("Step 1 : init config")
    print("============================================================")

    # สร้าง args คล้าย argparse
    args = SimpleNamespace(
        input_path=input_path,
        output_path=None,
        fidelity_weight=fidelity_weight,
        upscale=upscale,
        has_aligned=False,
        only_center_face=False,
        draw_box=False,
        detection_model='retinaface_resnet50',
        bg_upsampler=bg_upsampler,
        face_upsample=face_upsample,
        bg_tile=400,
        suffix=None
    )

    sys.argv = ['']  # ล้าง arg เพื่อไม่ให้ argparse เดิมสับสน

    print()
    print("============================================================")
    print("Step 2 : input & output")
    print("============================================================")

    # ------------------------ input & output ------------------------
    w = args.fidelity_weight
    input_video = False
    if args.input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        from basicsr.utils.video_util import VideoReader, VideoWriter
        input_img_list = []
        vidreader = VideoReader(args.input_path)
        image = vidreader.get_frame()
        while image is not None:
            input_img_list.append(image)
            image = vidreader.get_frame()
        audio = vidreader.get_audio()
        fps = vidreader.get_fps()    
        video_name = os.path.basename(args.input_path)[:-4]
        if not args.output_path:
            args.output_path = './output_videos'

        video_name = os.path.splitext(os.path.basename(args.input_path))[0]
        result_root = os.path.join(args.output_path, video_name)

        input_video = True
        vidreader.close()
    else: 
        raise RuntimeError("input should be mp4 file")

    if not args.output_path is None: # set output path
        result_root = args.output_path

    test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError('No input image/video is found...\n' 
            '\tNote that --input_path for video should end with .mp4|.mov|.avi')

    print()
    print("============================================================")
    print("Step 3 : set up background upsampler")
    print("============================================================")
    # ------------------ set up background upsampler ------------------
    if args.bg_upsampler == 'realesrgan':
        bg_upsampler = set_realesrgan(tile_size=args.bg_tile, upscale=args.upscale)  # ✅ ส่ง upscale ไปด้วย
    else:
        bg_upsampler = None


    print()
    print("============================================================")
    print("Step 4 : set up face upsampler")
    print("============================================================")
    # ------------------ set up face upsampler ------------------
    if args.face_upsample:
        if bg_upsampler is not None:
            face_upsampler = bg_upsampler
        else:
            face_upsampler = set_realesrgan()
    else:
        face_upsampler = None


    print()
    print("============================================================")
    print("Step 5 : set up CodeFormer restorer")
    print("============================================================")
    from urllib.request import urlretrieve
    import sys

    def download_with_progress(url, filename):
        def show_progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            percent = min(percent, 100)
            sys.stdout.write(f"\r📦 Downloading... {percent}%")
            sys.stdout.flush()
            if percent == 100:
                print(" ✅ Done.")
        
        urlretrieve(url, filename, reporthook=show_progress)

    net = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=['32', '64', '128', '256']
    ).to(device)

    ckpt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../pretrained_models/hallo2"))
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "net_g.pth")

    if not os.path.exists(ckpt_path):
        print("🔄 Downloading net_g.pth ...")
        try:
            download_with_progress(
                "https://huggingface.co/fudan-generative-ai/hallo2/resolve/main/hallo2/net_g.pth?download=true",
                ckpt_path
            )
        except Exception as e:
            print(f"\n❌ Failed to download net_g.pth: {e}")
            raise

    checkpoint = torch.load(ckpt_path)['params_ema']
    m, n = net.load_state_dict(checkpoint, strict=False)
    print("missing key: ", m)
    assert len(n) == 0
    net.eval()



    print()
    print("============================================================")
    print("Step 6 : set up FaceRestoreHelper")
    print("============================================================")
    # ------------------ set up FaceRestoreHelper -------------------
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    if not args.has_aligned: 
        print(f'Face detection model: {args.detection_model}')
    if bg_upsampler is not None: 
        print(f'Background upsampling: True, Face upsampling: {args.face_upsample}')
    else:
        print(f'Background upsampling: False, Face upsampling: {args.face_upsample}')

    face_helper = FaceRestoreHelper(
        args.upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model = args.detection_model,
        save_ext='png',
        use_parse=True,
        device=device)
    
    n = -1
    input_img_list = input_img_list[:n]
    length = len(input_img_list)

    overlay = 4
    chunk = 16
    idx_list = []

    i=0
    j=0
    while i < length and j < length:
        j = min(i+chunk, length)
        idx_list.append([i, j])
        i = j-overlay
        

    id_list = []

    video_name = os.path.splitext(os.path.basename(args.input_path))[0]
    result_root = os.path.join(args.output_path, video_name)
    os.makedirs(os.path.join(result_root, 'final_results'), exist_ok=True)

    # Step 6: read video frame
    input_img_list = read_input_images(args)
    print(f"input_img_list : {len(input_img_list)}")


    print()
    print("============================================================")
    print("Step 7 : start to processing - version 3")
    print("============================================================")

    final_result_dir = os.path.join(result_root, 'final_results')
    if os.path.exists(final_result_dir):
        shutil.rmtree(final_result_dir)
    os.makedirs(final_result_dir, exist_ok=True)

    total_frames = len(input_img_list)

    for frame_idx, img_path in enumerate(input_img_list):
        try:
            progress((frame_idx + 1) / total_frames, desc=f"Step 7: Frame {frame_idx + 1}")
            face_helper.clean_all()

            # Load image
            if isinstance(img_path, str):
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                print(f'  [{frame_idx+1}/{total_frames}] Processing: {os.path.basename(img_path)}')
            else:
                img = img_path
                print(f'  [{frame_idx+1}/{total_frames}] Processing: frame')

            # Face detection
            if args.has_aligned:
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_helper.is_gray = is_gray(img, threshold=10)
                face_helper.cropped_faces = [img]
            else:
                face_helper.read_image(img)
                num_faces = face_helper.get_face_landmarks_5(
                    only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5)
                print(f'    detect {num_faces} faces')
                face_helper.align_warp_face()

            # Prepare cropped faces
            crop_image = []
            for cropped_face in face_helper.cropped_faces:
                t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                crop_image.append(t.unsqueeze(0))

            # Run face restoration
            output = None
            try:
                if crop_image:
                    crop_tensor = torch.cat(crop_image, dim=0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output, _ = net.inference(crop_tensor, w=w, adain=True)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("CUDA OOM. Falling back to per-image inference.")
                    torch.cuda.empty_cache()
                    output_list = []
                    for t in crop_image:
                        try:
                            with torch.no_grad():
                                out, _ = net.inference(t.unsqueeze(0).to(device), w=w, adain=True)
                                output_list.append(out.cpu())
                        except Exception as e_img:
                            print(f"Skip face due to error: {e_img}")
                    if output_list:
                        output = torch.cat(output_list, dim=1).to(device)
                    else:
                        print(f"⚠️  Frame {frame_idx+1}: No valid outputs in fallback.")
                else:
                    raise e

            print("DEBUG: Face restoration completed.")

            # Add restored faces back
            if output is not None:
                for k in range(output.shape[1]):
                    face_output = output[:, k:k+1]
                    restored = tensor2img(face_output.squeeze_(1), rgb2bgr=True, min_max=(-1, 1)).astype('uint8')
                    try:
                        cropped_face = face_helper.cropped_faces[k]
                        face_helper.add_restored_face(restored, cropped_face)
                    except IndexError:
                        print(f"⚠️  Skipped saving restored face {k} due to mismatch.")
            else:
                face_helper.restored_faces = []

            # Background upscaling
            bg_img = None
            if bg_upsampler is not None:
                try:
                    bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0]
                except:
                    bg_img = None

            # Merge restored face back to original image
            try:
                face_helper.get_inverse_affine(None)
                if len(face_helper.restored_faces) != len(face_helper.affine_matrices):
                    print("Warning: Face mismatch (restored_faces != affine_matrices), using bg only")
                    restored_img = bg_img if bg_img is not None else img
                else:
                    restored_img = face_helper.paste_faces_to_input_image(
                        upsample_img_list=[bg_img],
                        draw_box=args.draw_box,
                        face_upsampler=face_upsampler if args.face_upsample else None
                    )[0]
            except Exception as e:
                print(f"Error processing frame {frame_idx+1}: {e}")
                restored_img = bg_img if bg_img is not None else img

            # Save image as running number
            outname = f"{frame_idx:05d}.png"
            save_path = os.path.join(final_result_dir, outname)
            if isinstance(restored_img, np.ndarray):
                imwrite(restored_img, save_path)
            else:
                print(f"⚠️  Skipped saving frame {frame_idx+1} because restored_img is not valid.")

        except Exception as err:
            print(f"⚠️  Failed to process frame {frame_idx+1}: {err}")
            continue


    print()
    print("============================================================")
    print("Step 8 : save enhanced video")
    print("============================================================")

    if input_video:
        print('🎞️ Video Saving...')

        final_result_dir = os.path.join(result_root, 'final_results')
        img_list = glob.glob(os.path.join(final_result_dir, '*.[jp][pn]g'))

        # แก้ปัญหาการจัดเรียงเฟรมผิดลำดับ เช่น xxx_9.png มาก่อน xxx_10.png
        def extract_frame_index(filename):
            basename = os.path.basename(filename)
            index_str = os.path.splitext(basename)[0].split('_')[-1]
            try:
                return int(index_str)
            except:
                return -1  # ถ้าแปลงไม่ได้ให้โยนไปท้าย

        img_list = sorted(img_list, key=extract_frame_index)

        if not img_list:
            raise RuntimeError("❌ No frames found in final_results/. Cannot generate video.")

        print(f"📸 Found {len(img_list)} frames in final_results/")

        if len(img_list) != length:
            print(f"⚠️ Warning: Frame count mismatch! expected={length}, found={len(img_list)}")

        sample_img = cv2.imread(img_list[0])
        height, width = sample_img.shape[:2]
        print(f"🖼️ Video resolution: {width}x{height}, FPS: {fps}")

        # ตั้งชื่อไฟล์ผลลัพธ์
        video_file_name = f"{video_name}_{args.suffix}.mp4" if args.suffix else f"{video_name}.mp4"
        save_restore_path = os.path.join(result_root, video_file_name)

        vidwriter = VideoWriter(save_restore_path, height, width, fps, audio)
        for idx, img_path in enumerate(img_list):
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Skipping unreadable frame: {img_path}")
                continue
            vidwriter.write_frame(img)
            if idx % 10 == 0 or idx == len(img_list) - 1:
                print(f"🧵 Writing frame {idx+1}/{len(img_list)}...")

        vidwriter.close()
        print(f"✅ Video saved at: {save_restore_path}")

        # ลบโฟลเดอร์ final_results
        try:
            shutil.rmtree(final_result_dir)
            print(f"🧹 Deleted temp folder: {final_result_dir}")
        except Exception as e:
            print(f"⚠️ Failed to delete {final_result_dir}: {e}")

    print(f"\n📁 All results are saved in: {result_root}")

    # คืนค่าให้ Gradio
    return save_restore_path, f"✅ Done! Video saved at: {save_restore_path}"



def create_video_sr_interface():
    with gr.Blocks(title="Video Super Resolution") as demo:
        gr.Markdown("## 🎥 Video Super Resolution with Face Restoration")

        with gr.Row():
            input_video = gr.Video(label="Input Video (.mp4)")
            output_video = gr.Video(label="Output Video")

        with gr.Row():
            fidelity_weight = gr.Slider(0, 1, value=1.0, step=0.1, label="Fidelity Weight (e.g. 1)")
            upscale = gr.Radio([2, 4, 8], value=4, label="Upscale Factor")  # ✅ เพิ่ม x8

        with gr.Row():
            bg_upsampler = gr.Dropdown(["None", "realesrgan"], value="realesrgan", label="Background Upsampler")
            face_upsample = gr.Checkbox(label="Use Face Upsample", value=True)

        status_text = gr.Textbox(label="Status", interactive=False)
        generate_button = gr.Button("Run Video Super Resolution")

        generate_button.click(
            fn=process_video_sr,
            inputs=[input_video, fidelity_weight, upscale, bg_upsampler, face_upsample],
            outputs=[output_video, status_text]
        )

    return demo



if __name__ == "__main__":
    import webbrowser

    demo = create_video_sr_interface()

    # เปิด browser พร้อมธีม dark โดยกำหนด URL
    url = "http://127.0.0.1:7861/?__theme=dark"
    webbrowser.open(url)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        debug=True,
        share=False,  # ปิด share หากไม่ต้องการออนไลน์
        prevent_thread_lock=True  # ป้องกันไม่ให้บล็อค main thread เมื่อใช้ webbrowser
    )


