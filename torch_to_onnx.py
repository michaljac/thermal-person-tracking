from ultralytics import YOLO
import os
import cv2

def onnx_export(torch_path, imgsz, opset=12, half=False, device='cpu'):

    model = YOLO(torch_path)
    print(model.names) 

    onnx_path = model.export(
        format='onnx',
        dynamic=False,
        simplify=False,
        opset=opset,
        half=half,
        imgsz=imgsz,    # (H,W)
        device=device,  # 'cpu' / '0'
        nms=False,
        verbose=False
        )

    print('Onnx Converted model seved path:', onnx_path)
    os.chmod(onnx_path, 0o777)
    print()

    return onnx_path

if __name__ == '__main__':

    torch_path = 'models/best_uav_fast.pt'
    # input_size = 1024
    imgsz = 512 # 1024 / (608, 1024)  # Onnx input size: INT or [H,W]

    # Convert
    onnx_path = onnx_export(torch_path, imgsz, opset=12, half=False, device='cpu')

    # Test Converted Onnx model Inference
    from onnx_detector import OnnxDetector_Yolov8
    image_path = 'data/MAX_0014_1000x750_frame2397.jpg' 
    output_dir = 'results'
    filename, ext = os.path.splitext(os.path.basename(image_path))
    out_file = filename + '_onnx' + ext
    out_path = os.path.join(output_dir, out_file)

    detector = OnnxDetector_Yolov8(
    model_path=onnx_path,
    class_names=['person'],
    conf_thresh=0.2,
    iou_thresh=0.45,
    use_gpu=False   
    )

    detector.infer_time_test(batch=1, n=100)

    img = cv2.imread(image_path)
    detections = detector.infer(img)
    frame_out = detector.draw(img.copy(), detections)

    cv2.imwrite(out_path, frame_out)
    print('output image saved as: ', out_path)