def create_model():

    import google.protobuf
    from google.protobuf.internal import api_implementation
    print("protobuf path: ", google.protobuf.__file__) 
    print("protobuf type: ", api_implementation.Type())
    print("protobuf Version: ", api_implementation.Version())
    import traceback
    import logging
    logging.basicConfig(filename='/home/jkuo/maplab_interface.log', filemode='w',level=logging.NOTSET)
    logging.debug('This message should go to the log file')
    logging.info('So should this')
    logging.warning('And this, too')
    try:
        logging.debug('in try')
        import os
        import sys
        import skimage.io
        logging.debug('can import standard library')
        print("sys version: ", sys.version)
        print("prefix: ", sys.prefix)
        print("exec_prefix: ", sys.exec_prefix)
        print("executable: ", sys.executable)
        print("api_version: ", sys.api_version)
        # Root directory of the project
        # ROOT_DIR = os.path.abspath("../") // = normpath(join(os.getcwd(), path))
        ROOT_DIR = "/home/jkuo/work/Mask_RCNN"
        print("Root dir in the interface file is: ", ROOT_DIR)
        # sys.path.append(ROOT_DIR)
        # import tensorflow as tf
        # print("end")
        # sys.stdout.flush()
        # sys.path = []
        # sys.path = ['', '/home/jkuo/maplab_ws/devel/lib/python2.7/dist-packages', '/opt/ros/kinetic/lib/python2.7/dist-packages', '/usr/local', '/usr/lib/python2.7', '/usr/lib/python2.7/plat-x86_64-linux-gnu', '/usr/lib/python2.7/lib-tk', '/usr/lib/python2.7/lib-old', '/usr/lib/python2.7/lib-dynload', '/home/jkuo/.local/lib/python2.7/site-packages', '/usr/local/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages', '/usr/lib/python2.7/dist-packages/PILcompat', '/usr/lib/python2.7/dist-packages/gtk-2.0', '/usr/lib/python2.7/dist-packages/wx-3.0-gtk2']
        # logging.debug('the sys path: %s', sys.path)
        # import tensorflow as tf
        # logging.debug('import tensorflow success')
        # Import Mask RCNN
        import mrcnn.model as modellib
        logging.debug('can import mrcnn.model')


        # # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

                

        # # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # logging.debug('the model dir: %s', COCO_MODEL_PATH)
        from mrcnn.config import Config
        # class InferenceConfig(coco.CocoConfig):
        class InferenceConfig(Config):
            NAME = "coco"
            NUM_CLASSES = 1 + 80
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()

        logging.debug('the config: %s', config)

        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        model.load_weights(COCO_MODEL_PATH, by_name=True)

        # testing
        print(model)
        IMAGE_DIR = os.path.join(ROOT_DIR, "images")
        print("image dir: ", IMAGE_DIR)
        file_names = next(os.walk(IMAGE_DIR))[2]
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[10]))
        print("image name: ",image)
        # Run detection
        results = model.detect([image], verbose=1)

        print(results)

        return model

        # return 1
    except Exception as e:
        logging.debug('in error')
        logging.error(traceback.format_exc())
    
    