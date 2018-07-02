import pandas as pd
from scipy import misc
import argparse
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import scipy.misc
import os, sys
import shutil
import fnmatch
import math
import random, shutil
from PIL import Image
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras import optimizers, initializers
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from retrain_yolo import process_data,process_data_pil,process_data_pil_wide,get_classes,get_anchors,get_detector_mask,train,draw
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes,preprocess_true_boxes_true_box, yolo_loss, yolo_body, yolo_eval


def predict_any(sess , model, image_file, anchors, class_names, max_boxes, score_threshold, iou_threshold):
    
    # Get head of model
    yolo_outputs_ = yolo_head(model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    
    # Preprocess your image
    model_image_size =  model.inputs[0].get_shape().as_list()[-3:-1]
    image, image_data = preprocess_image(image_file, model_image_size =  model_image_size )
    
    img=plt.imread(image_file)
    img_shape_ = img.shape[0:2]
    print(  "Reshaping input image "+str( img_shape_)  +" to model input shape "+str(model_image_size)  )
    if img_shape_[0]>img_shape_[1]:
        print( "Wrong input size ",str( img_shape_), " Exiting"  )
        return (0,0,0)
    # Get the Tensors
    boxes, scores, classes = yolo_eval(yolo_outputs_, [float(i) for i in list(img_shape_)],
                max_boxes,
              score_threshold,
              iou_threshold)  

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model.input: image_data,
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file.split('/')[-1]), quality=90)
    # Display the results in the notebook
    plt.figure()

    output_image = scipy.misc.imread(os.path.join("out", image_file.split('/')[-1]))
    plt.imshow(output_image)

    return out_scores, out_boxes, out_classes

# Wrap the Yolo model with other model for training: You can select how to wrpait and if you wnat to do transfer learning
# Create model around yolo model 
#     Use freeze_body for doing transfer learning on 1st training stage 
def create_model(anchors, class_names, load_pretrained=True, freeze_body=True, regularization_rate = 0.01):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)
    
    # Implement regularization
    if regularization_rate: # if we want regularization
        for layer in topless_yolo.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = regularization_rate
    
    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model

def create_model_wide(sess,anchors, class_names, load_pretrained=True, freeze_body=True, regularization_rate = 0.01,
                      initialize_weights = False):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (13, 19, 5, 1)
    matching_boxes_shape = (13, 19, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 608, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)
    
    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)

    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)
    
    model_body = Model(image_input, final_layer)
    
    # Implement regularization
    if regularization_rate: # if we want regularization
        for layer in model_body.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = regularization_rate
    
    # Implement initialize_weights
    if initialize_weights: # if we want initialize
        for layer in model_body.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = initializers.get("glorot_uniform")
                layer.kernel.initializer.run(session=sess)
                
#     print(model_body.output)
    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model

def get_batch(list_filenames, batch_size,boxes_dir, class_idx,classes_path,anchors_path): 
    # Get anchors and classes names
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    Img_db = pd.read_csv(boxes_dir, header = 0)
    while True:
        for batches in range(len(list_filenames) // batch_size):
            images_list = []
            boxes_list = []
            image_data = None
            boxes = None
            for image_sample in list_filenames[batches*batch_size:min(len(list_filenames),(batches+1)*batch_size)]:
            
#                 images_list.append( mpimg.imread(image_sample)  )
                images_list.append( Image.open( image_sample )  )
                                   
                # Write the labels and boxes
                # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
                labels_boxes = []
                #     print(Img_db[Img_db['image_filename']==image_sample.split("/")[-1]].as_matrix())
                for box_matched in Img_db[Img_db['image_filename']==image_sample.split("/")[-1]].as_matrix():
                    labels_boxes.append( [class_idx[box_matched[-2]], *box_matched[2:6]] )
                boxes_list.append(np.asarray(labels_boxes))
                
                # Check if image is fliped and portrait it
                if images_list[-1].width<images_list[-1].height:
                    images_list[-1] = images_list[-1].transpose(Image.TRANSPOSE)
                    for i,boxs in enumerate(boxes_list[-1]) :
                        boxes_list[-1][i] =  [  boxs[0], boxs[2],boxs[1],boxs[4],boxs[3]  ]
                        
                #print(image_sample)
            ### Preprocess the data: get images and boxes
            # get images and boxes
            image_data, boxes = process_data_pil(images_list, boxes_list)
        
            ### Precompute detectors_mask and matching_true_boxes for training
            # Precompute detectors_mask and matching_true_boxes for training
            detectors_mask = [0 for i in range(len(boxes))]
            matching_true_boxes = [0 for i in range(len(boxes))]
            for i, box in enumerate(boxes):
                detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

            detectors_mask = np.array(detectors_mask)
            matching_true_boxes = np.array(matching_true_boxes)
            
            # yield x_batch, y_batch
            yield ( [image_data, boxes, detectors_mask, matching_true_boxes], np.zeros(len(image_data)) )
            
def get_batch_wide(list_filenames, batch_size,boxes_dir, class_idx,classes_path,anchors_path,true_boxes_flag=False): 
    # Get anchors and classes names
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    Img_db = pd.read_csv(boxes_dir, header = 0)
    while True:
        for batches in range(len(list_filenames) // batch_size):
            images_list = []
            boxes_list = []
            image_data = None
            boxes = None
            for image_sample in list_filenames[batches*batch_size:min(len(list_filenames),(batches+1)*batch_size)]:
            
#                 images_list.append( mpimg.imread(image_sample)  )
                images_list.append( Image.open( image_sample )  )

                # Write the labels and boxes
                # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
                labels_boxes = []
                #     print(Img_db[Img_db['image_filename']==image_sample.split("/")[-1]].as_matrix())
                for box_matched in Img_db[Img_db['image_filename']==image_sample.split("/")[-1]].as_matrix():
                    labels_boxes.append( [class_idx[box_matched[-2]], *box_matched[2:6]] )
                boxes_list.append(np.asarray(labels_boxes))
                
                
                # Check if image is fliped and portrait it
                if images_list[-1].width<images_list[-1].height:
                    images_list[-1] = images_list[-1].transpose(Image.TRANSPOSE)
                    for i,boxs in enumerate(boxes_list[-1]) :
                        boxes_list[-1][i] =  [  boxs[0], boxs[2],boxs[1],boxs[4],boxs[3]  ]
                    
                #print(image_sample)
            ### Preprocess the data: get images and boxes
            # get images and boxes
            image_data, boxes = process_data_pil_wide(images_list, boxes_list)
#             print(image_data.shape)
            ### Precompute detectors_mask and matching_true_boxes for training
            # Precompute detectors_mask and matching_true_boxes for training
            detectors_mask = [0 for i in range(len(boxes))]
            matching_true_boxes = [0 for i in range(len(boxes))]
            
            if true_boxes_flag:
                for i, box in enumerate(boxes):
                    detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes_true_box(box, anchors, [416, 608])
            else:
                for i, box in enumerate(boxes):
                    detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 608])
                    
            
            detectors_mask = np.array(detectors_mask)
            matching_true_boxes = np.array(matching_true_boxes)
            
            # yield x_batch, y_batch
            yield ( [image_data, boxes, detectors_mask, matching_true_boxes], np.zeros(len(image_data)) )
def iou(box1, box2): 
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    ### START CODE HERE ### (≈ 5 lines)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1)*(yi2 - yi1)
    ### END CODE HERE ###    

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ### START CODE HERE ### (≈ 3 lines)
    box1_area = (box1[3] - box1[1])*(box1[2]- box1[0])
    box2_area = (box2[3] - box2[1])*(box2[2]- box2[0])
    union_area = (box1_area + box2_area) - inter_area
    ### END CODE HERE ###
    
    # compute the IoU
    ### START CODE HERE ### (≈ 1 line)
    iou = float(inter_area) / float(union_area)
    ### END CODE HERE ###

    return iou

# Batch mAP evaluation
# Batch mAP evaluation
def mAP_eval(sess , model, image_files,boxes_dir, anchors,class_idx, class_names, max_boxes, score_threshold, iou_threshold=0.5, 
             iou_eval_threshold = 0.5, plot_compare = False):
    # Get head of model
    yolo_outputs_ = yolo_head(model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    # Get the database
    Img_db = pd.read_csv(boxes_dir, header = 0)
    
    # Get input image size
    img=plt.imread(image_files[0])
    img_shape_ = img.shape[0:2]
#     print(  "Reshaping input image "+str( img_shape_)  +" to model input shape "+str(model_image_size)  )
    
    # Get model input size
    model_image_size =  model.inputs[0].get_shape().as_list()[-3:-1]
    
    # Get the Tensors
    boxes, scores, classes = yolo_eval(yolo_outputs_, [float(i) for i in list(img_shape_)],
                max_boxes,
              score_threshold,
              iou_threshold)


    positive_detections = 0
    positive_samples = 0
    true_positives = 0
    image = None
    image_gt = None
    for image_file in image_files: # Loop over all the files
        ## Get models output
        # Preprocess your image
        image, image_data = preprocess_image(image_file, model_image_size =  model_image_size )
        if image.width<image.height:
            print( "Wrong input size, Passing"  )
            pass
        # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
        out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    model.input: image_data,
                    input_image_shape: [image_data.shape[2], image_data.shape[3]],
                    K.learning_phase(): 0
                })
        # output boxes/class matrix
        labels_boxes_pred = np.insert(out_boxes,0,out_classes.T,axis=1) 
        
        ## Get dataset labels/boxes
        # Write the labels and boxes
        labels_boxes = []

        for box_matched in Img_db[Img_db['image_filename']==image_file.split("/")[-1]].as_matrix():
            labels_boxes.append( [class_idx[box_matched[-2]], box_matched[3],box_matched[2],box_matched[5],box_matched[4] ] )
        labels_boxes_ground = np.asarray(labels_boxes)
        
        if plot_compare:
            if (labels_boxes_ground.shape[0]):
                # Get image
                image_gt, _ = preprocess_image(image_file, model_image_size =  model_image_size )
                # Plot in comparison: Ground truth
                print(' Ground truth')
                # Generate colors for drawing bounding boxes.
                colors = generate_colors(class_names)
                # Draw bounding boxes on the image file
    #             print(out_classes,labels_boxes_ground[:,0].astype(int) )
                draw_boxes(image_gt, np.ones(labels_boxes_ground.shape[0]), labels_boxes_ground[:,1:], labels_boxes_ground[:,0].astype(int), class_names, colors)
                # Save the predicted bounding box on the image
                image_gt.save(os.path.join("out_gt", image_file.split('/')[-1]), quality=90)
                # Display the results in the notebook
                plt.subplot(121)
                output_image = scipy.misc.imread(os.path.join("out_gt", image_file.split('/')[-1]))
                plt.imshow(output_image)
                plt.title(' Ground truth')
#                 plt.show()

            # Plot in comparison: Detection
            print(' Detection')
            # Generate colors for drawing bounding boxes.
            colors2 = generate_colors(class_names)
            # Draw bounding boxes on the image file
            draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors2)
            # Save the predicted bounding box on the image
            image.save(os.path.join("out", image_file.split('/')[-1]), quality=90)
            # Display the results in the notebook
            plt.subplot(122)
            output_image2 = scipy.misc.imread(os.path.join("out", image_file.split('/')[-1]))
            plt.imshow(output_image2)
            plt.title(' Prediction')
            plt.show()
        
            # control
            input("Press a Enter to continue...")
            
        ## Evaluation of all outputs vs. ground-data
        for i,preds in enumerate(labels_boxes_pred):
            for j,grounds in enumerate(labels_boxes_ground):
                if(  iou(preds[1:], grounds[1:]) >= iou_eval_threshold and preds[0] == grounds[0] ):
                    true_positives += 1
                    labels_boxes_pred[i][0]=100
                    labels_boxes_ground[j][0]=200

        # Precision and recall
        positive_detections += labels_boxes_pred.shape[0]
        positive_samples += labels_boxes_ground.shape[0]
        
                
#     print(true_positives,positive_samples,positive_detections)
    
    if(positive_detections!=0 and positive_samples!=0 ):
        precision =  float(true_positives) / float(positive_detections)
        recall =  float(true_positives) / float(positive_samples)

    print( " mean precision = ",precision," , mean recall =  ",recall )
    print( " Final F1 score =  ",2*precision*recall/(precision+recall) )

def nexar_eval_test(sess , model, image_files, boxes_dir, anchors,class_idx, class_names, max_boxes, score_threshold,
              iou_threshold=0.5, plot_result = False ):
    ## Evaluate all images in the the image_files list (Test images) and save the results to an Excel files with results
    # Resturns a Pandas Dataframe as a table with all the outputs boxes, confidences, etc
    
    # Define output dict
    results = { 'image_filename': [],'x0':[],'y0':[],'x1':[],'y1':[],'label':[],'confidence':[] }
    # Get head of model
    yolo_outputs_ = yolo_head(model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    # Get the database
#     Test_img_db = pd.read_csv(boxes_dir, header = 0)
    
    # Get input image size
    img=plt.imread(image_files[0])
    img_shape_ = img.shape[0:2]
    
    # Get model input size
    model_image_size =  model.inputs[0].get_shape().as_list()[-3:-1]
    
    # Get the Tensors
    boxes, scores, classes = yolo_eval(yolo_outputs_, [float(i) for i in list(img_shape_)],
                max_boxes,
              score_threshold,
              iou_threshold)

    for image_file in image_files: # Loop over all the files
        ## Get models output
        # Preprocess your image
        image, image_data = preprocess_image(image_file, model_image_size =  model_image_size )
        # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
        out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    model.input: image_data,
                    input_image_shape: [image_data.shape[2], image_data.shape[3]],
                    K.learning_phase(): 0
                })
        if plot_result:
            # Plot in comparison: Detection
            # Generate colors for drawing bounding boxes.
            colors = generate_colors(class_names)
            # Draw bounding boxes on the image file
            draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
            # Save the predicted bounding box on the image
            image.save(os.path.join("out", image_file.split('/')[-1]), quality=90)
            # Display the results in the notebook
            output_image = scipy.misc.imread(os.path.join("out", image_file.split('/')[-1]))
            plt.imshow(output_image)
            plt.show()
            # control
            input("Press a Enter to continue...")
            
        # Save to results AND Swap x and y dimensions: they are ok for ploting but not for submiting
        # order is x0,y0,x1,y1
        for i in range(0,out_boxes.shape[0]):
            results['image_filename'].append( image_file.split('/')[-1] )
            results['x0'].append( out_boxes[i,1] )
            results['y0'].append( out_boxes[i,0] )
            results['x1'].append( out_boxes[i,3] )
            results['y1'].append( out_boxes[i,2] )
            results['label'].append(  class_names[ out_classes[i] ] )
            results['confidence'].append(  out_scores[i] )
    # To Pandas and Excel
    df = pd.DataFrame(data=results)
    df=df[[ 'image_filename','x0','y0','x1','y1','label','confidence' ]]
    
    return df
def preprocess_true_boxes_true_box(true_boxes, anchors, image_size):
    """Find detector in YOLO where ground truth box should appear.

    Parameters
    ----------
    true_boxes : array
        List of ground truth boxes in form of relative x, y, w, h, class.
        Relative coordinates are in the range [0, 1] indicating a percentage
        of the original image dimensions.
    anchors : array
        List of anchors in form of w, h.
        Anchors are assumed to be in the range [0, conv_size] where conv_size
        is the spatial dimension of the final convolutional features.
    image_size : array-like
        List of image dimensions in form of h, w in pixels.

    Returns
    -------
    detectors_mask : array
        0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
        that should be compared with a matching ground truth box.
    matching_true_boxes: array
        Same shape as detectors_mask with the corresponding ground truth box
        adjusted for comparison with predicted parameters at training time.
    """
    height, width = image_size
    num_anchors = len(anchors)
    # Downsampling factor of 5x 2-stride max_pools == 32.
    # TODO: Remove hardcoding of downscaling calculations.
    assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
    conv_height = height // 32
    conv_width = width // 32
#     print(conv_width,conv_height)
    num_box_params = true_boxes.shape[1]
    detectors_mask = np.zeros(
        (conv_height, conv_width, num_anchors, 1), dtype=np.float32)
    matching_true_boxes = np.zeros(
        (conv_height, conv_width, num_anchors, num_box_params),
        dtype=np.float32)
#     print(detectors_mask.shape)
    for box in true_boxes:
        # scale box to convolutional feature spatial dimensions
        box_class = box[4:5]
        box = box[0:4] * np.array(
            [conv_width, conv_height, conv_width, conv_height])
        i = np.floor(box[1]).astype('int')
        j = np.floor(box[0]).astype('int')
        best_iou = 0
        best_anchor = 0
        for k, anchor in enumerate(anchors):
            # Find IOU between box shifted to origin and anchor box.
            box_maxes = box[2:4] / 2.
            box_mins = -box_maxes
            anchor_maxes = (anchor / 2.)
            anchor_mins = -anchor_maxes
            
            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[0] * intersect_wh[1]
            box_area = box[2] * box[3]
            anchor_area = anchor[0] * anchor[1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            if iou > best_iou:
                best_iou = iou
                best_anchor = k

        if best_iou > 0:
            detectors_mask[i, j, best_anchor] = 1
            adjusted_box = np.array(
                [
                    box[0] - j, box[1] - i,
#                     np.log(box[2] / anchors[best_anchor][0]),
#                     np.log(box[3] / anchors[best_anchor][1]), box_class
                    box[2],
                    box[3], box_class
                ],
                dtype=np.float32)
            matching_true_boxes[i, j, best_anchor] = adjusted_box
    return detectors_mask, matching_true_boxes
