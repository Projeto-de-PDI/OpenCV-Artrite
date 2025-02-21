from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
import os


input_dir = "imagens_originais/"

output_base_dir = "imagens_aumentadas/"
os.makedirs(output_base_dir, exist_ok=True)

augmentations = {
    "rotation": ImageDataGenerator(rotation_range=30),
    "width_shift": ImageDataGenerator(width_shift_range=0.15),
    "height_shift": ImageDataGenerator(height_shift_range=0.15),
    "zoom": ImageDataGenerator(zoom_range=0.15),
    "horizontal_flip": ImageDataGenerator(horizontal_flip=True)
}

num_images_per_original = 10  

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(input_dir, filename)
        img = load_img(img_path)  
        img_array = img_to_array(img) 
        img_array = img_array.reshape((1,) + img_array.shape)  

        print(f"Processando: {filename}")

        for aug_name, datagen in augmentations.items():
            output_dir = os.path.join(output_base_dir, aug_name) 
            os.makedirs(output_dir, exist_ok=True)

            i = 0
            for batch in datagen.flow(img_array, batch_size=1):
                new_image_path = os.path.join(output_dir, f"{filename.split('.')[0]}_{aug_name}_{i}.jpg")
                save_img(new_image_path, batch[0]) 
                print(f"Imagem salva: {new_image_path}")
                i += 1
                if i >= num_images_per_original:
                    break  

print("Concluído")
