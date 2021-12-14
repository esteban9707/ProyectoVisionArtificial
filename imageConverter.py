import base64
def img_to_base64(image_path):
    image = open(image_path, 'rb')
    image_read = image.read()
    image_64_encode = base64.b64encode(image_read)
    return image_64_encode.decode('utf-8')


def base64_to_img(stringbase64, id):
    img_path = f'app/images/{id}.jpeg'
    decodeit = open(img_path, 'wb')
    decodeit.write(base64.b64decode(stringbase64))
    decodeit.close()
    return img_path
