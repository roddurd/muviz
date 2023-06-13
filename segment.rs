use image::GenericImage;
use image::Pixel;

fn overlay(background: &mut image::DynamicImage, foreground: &image::DynamicImage, x: u32, y: u32) {
    for (i, j, pixel) in foreground.pixels() {
        let (red, green, blue, alpha) = pixel.channels4();
        if alpha != 0 { // If not transparent
            background.put_pixel(x + i, y + j, pixel);
        }
    }
}

fn image_to_tensor(img: &image::DynamicImage) -> Tensor {
    let img = img.to_rgb8();
    let (width, height) = img.dimensions();
    let arr: Vec<_> = img.pixels().flat_map(|p| p.channels().iter().copied()).collect();

    Tensor::of_slice(&arr)
        .reshape(&[height as i64, width as i64, 3])
        .permute(&[2, 0, 1])
        .to_kind(Kind::Float) / 255.0
}

fn tensor_to_image(tensor: &Tensor) -> image::DynamicImage {
    let tensor = tensor
        .squeeze()
        .to_kind(Kind::Uint8);  // Make sure to use the appropriate kind
    let (height, width) = (tensor.size()[0], tensor.size()[1]);
    let vec: Vec<u8> = tensor.into();
    image::ImageBuffer::from_vec(width as u32, height as u32, vec).unwrap().into()
}


// Assume the new background is in the file "background.png"
let mut new_background = image::open("background.png").unwrap().to_rgba8();

while let Ok((stream, packet)) = decoder.read() {
    if stream.index() == video_stream_index {
        // Decode and process the frame...
        let mut decoded = ffmpeg_next::software::frame::Video::empty();
        let mut codec = video_stream.codec().decoder().video().unwrap();
        codec.decode(&packet, &mut decoded).unwrap();

        let raw_frame = codec.receive_frame().unwrap();
        let (width, height) = (raw_frame.width(), raw_frame.height());

        // Convert the frame to an ImageBuffer
        let frame_image = ImageReader::new(Cursor::new(raw_frame.data())).decode().unwrap();
        let frame_tensor = image_to_tensor(&frame_image);

        // Apply the model to the image
        let segmented_tensor = unet_model.forward_t(&frame_tensor.unsqueeze(0), true);
        
        // Convert the segmentation result into an image
        let segmented_image = tensor_to_image(&segmented_tensor);
        
        // Create a mask where 1s represent the foreground and 0s represent the background
        let mask = segmented_image.map_pixels(|x, y, pixel| {
            if pixel[0] > 128 { // Assuming grayscale image where white represents the foreground
                image::Rgba([255u8; 4])
            } else {
                image::Rgba([0u8, 0u8, 0u8, 0u8])
            }
        });
        
        // Multiply the frame image by the mask to get the foreground
        let foreground = frame_image.map_pixels(|x, y, pixel| {
            let mask_pixel = mask.get_pixel(x, y);
            if mask_pixel[0] > 128 { // If the mask pixel is white
                pixel
            } else {
                image::Rgba([0u8, 0u8, 0u8, 0u8])
            }
        });
        
        // Overlay the foreground onto the new background
        overlay(&mut new_background, &foreground, 0, 0);
    }
}
