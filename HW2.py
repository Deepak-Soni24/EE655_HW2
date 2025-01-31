import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_integral_image(image):
    if image.dtype not in [np.uint8, np.float32]:
        image = image.astype(np.float32) 
    
    integral_img = cv2.integral(image) 
    return integral_img

def apply_filter(integral_img, filter_mask, step_size):
    h, w = filter_mask.shape
    img_h, img_w = integral_img.shape[0], integral_img.shape[1]  
    responses = np.zeros((img_h - h - 1, img_w - w - 1))  

    for y in range(0, img_h - h - 1, step_size):
        for x in range(0, img_w - w - 1, step_size):
            white_sum = np.sum(filter_mask == 1)  
            gray_sum = np.sum(filter_mask == -1) 

            sum_region = (integral_img[y + h, x + w] - integral_img[y, x + w] 
                          - integral_img[y + h, x] + integral_img[y, x])
            
            if white_sum - gray_sum == 0:
                response = sum_region
            else:
                response = sum_region / (white_sum - gray_sum)  

            responses[y, x] = response
    
    return responses

def visualize_response(response_map, filter_index):
    min_val, max_val = response_map.min(), response_map.max()

    if max_val == min_val:  
        response_map = np.ones_like(response_map) * 127  
    else:
        response_map = (response_map - min_val) / (max_val - min_val) * 255

    response_map = response_map.astype(np.uint8)
    filename = f"response_filter_{filter_index}.png"
    cv2.imwrite(filename, response_map)
    print(f"Saved response map: {filename}")

    plt.figure(figsize=(6, 6))
    plt.imshow(response_map, cmap='gray')
    plt.title(f"Response Map for Filter {filter_index}")
    plt.axis("off")
    plt.show()

def main():
    image = cv2.imread('iitk.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found!")
        return
    
    binary_img = np.where(image > 127, 1, -1).astype(np.float32)
    
    integral_img = compute_integral_image(binary_img)
    
    filters = [
        np.array([[ -1,  -1, 1, 1],
                  [  -1, -1, 1, 1],
                  [ 1,  1, -1, -1],
                  [ 1, 1, -1, -1]]),
        
        np.array([[ -1, -1, -1, -1],
                  [ -1, -1, -1, -1],
                  [ 1, 1, 1, 1],
                  [ 1, 1, 1, 1]]),
        
        np.array([[ -1,  -1, 1, 1],
                  [  -1,  -1, 1, 1],
                  [ -1,  -1, 1, 1],
                  [-1,  -1, 1, 1]]),
        
        np.array([[  1, 1,  -1, -1],
                  [ 1, 1,  -1, -1],
                  [  -1, -1,  1, 1],
                  [-1, -1,  1, 1]]),
        
        np.array([[ -1, -1, -1, -1],
                  [  -1, -1, -1, -1],
                  [ -1, -1, 1, 1],
                  [-1, -1, 1, 1]]),
        
        np.array([[  1, 1,  1, 1],
                  [ 1,  -1, -1, -1],
                  [  1,  -1, -1, -1],
                  [1,  -1, -1, -1]])
    ]
    
    step_size = min([f.shape[0] for f in filters]) // 2  

    for i, filter_mask in enumerate(filters):
        response_map = apply_filter(integral_img, filter_mask, step_size)
        visualize_response(response_map, i+1)

if __name__ == "__main__":
    main()
