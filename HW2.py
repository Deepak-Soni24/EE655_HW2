import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_integral_image(image):
    img_h, img_w = image.shape
    integral_img = np.zeros((img_h + 1, img_w + 1), dtype=np.int64)
    
    for i in range(1, img_h + 1):
        for j in range(1, img_w + 1):
            integral_img[i, j] = (image[i - 1, j - 1] + integral_img[i - 1, j] +
                                   integral_img[i, j - 1] - integral_img[i - 1, j - 1])
    
    print("Integral Image Sample:")
    print(integral_img[:10, :10]) 
    
    return integral_img

def apply_filter(integral_img, filter_mask, filter_index):
    h, w = filter_mask.shape
    img_h, img_w = integral_img.shape[0] - 1, integral_img.shape[1] - 1  
    responses = np.zeros((img_h - h, img_w - w), dtype=np.int32)
    
    for y in range(img_h - h):
        for x in range(img_w - w):
            white_sum = np.sum(filter_mask == 1)
            black_sum = np.sum(filter_mask == -1)

            sum_region = (integral_img[y + h, x + w] - integral_img[y, x + w]
                          - integral_img[y + h, x] + integral_img[y, x])
            
            if white_sum - black_sum != 0:
                response = sum_region / (white_sum - black_sum)  
            else:
                response = sum_region  

            responses[y, x] = response
    
    print(f"Integral Image Sample for Filter {filter_index}:")
    print(responses[:10, :10])  
    return responses


def visualize_response(response_map, filter_index):
    print(f"Filter {filter_index} - Min: {response_map.min()}, Max: {response_map.max()}")
    response_map = cv2.normalize(response_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    filename = f"response_filter_{filter_index}.png"
    cv2.imwrite(filename, response_map)
    plt.imshow(response_map, cmap='gray')
    plt.title(f"Response Map for Filter {filter_index}")
    plt.axis("off")
    plt.show()

def main():
    image = cv2.imread('iitk.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found!")
        return
    
    integral_img = compute_integral_image(image)
    
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
    
    for i, filter_mask in enumerate(filters):
        response_map = apply_filter(integral_img, filter_mask, i+1)
        visualize_response(response_map, i+1)

if __name__ == "__main__":
    main()
