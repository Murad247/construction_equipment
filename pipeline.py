import tensorflow as tf, os, glob, random


def make_pipeline_multy_label(path, batch_size=256, shuffle=True, class_names=None, labels=True, par=False, im_size=(256, 256, 3)):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    W, H, D = im_size[0], im_size[1], im_size[2]
    if labels == True:
        addrs = glob.glob(path + str('*\\*.jpg' if path[-2:] == '\\' else '\\*\\*.jpg'))
    else:
        addrs = glob.glob(path + str('*.jpg' if path[-2:] == '\\' else '\\*.jpg'))
       
        if par == True:
            addrs2 = [int(name.split('\\')[-1].split('.')[0]) for name in addrs]
            addrs2 = sorted(addrs2)
            addrs = [path + f'\\{i}.jpg' for i in addrs2]

    if class_names == None:
        class_names = ['truck_crane',
                       'car', 
                       'excavator',
                       'human',
                       'dump_truck',
                       'mining_loader',
                       'rink',
                       'bulldozer']
    
    if shuffle:
        random.shuffle(addrs)
        
    def parse_image(filename):
        parts = tf.strings.split(filename, os.path.sep)
        
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=D)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [W, H])
        if labels == True:
            label = []
            for i in range(len(class_names)):
                label.append(1 if parts[-2] == class_names[i] else 0)
            return image, label
        else:
            return image

    def create_dataset(file_list):  
        ds=tf.data.Dataset.from_tensor_slices(file_list)  
        ds=ds.map(parse_image, num_parallel_calls=AUTOTUNE)
        ds=ds.batch(batch_size)
        ds = ds.prefetch(AUTOTUNE)
        return ds
    
    return create_dataset(addrs)
