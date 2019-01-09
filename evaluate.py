from . import transform
import numpy as np
import tensorflow as tf
from .utils import save_img, get_img  #exists and list_files not used
from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect as redirect
img_dict={}
BATCH_SIZE = 1  # use size 1 for 1 image (also avoids OOM error)
DEVICE = '/cpu'  # /gpu:0


# get img_shape
def ffwd(request,data_in, response, checkpoint, device_t='/cpu', batch_size=1):
    # assert len(paths_out) > 0
    # is_paths = type(data_in[0]) == str
    # if is_paths:
    #     assert len(data_in) == len(paths_out)
    #     img_shape = get_img(data_in[0]).shape
    # else:  # this case is when data_in is an img file!!! wtf is X???
    #     assert data_in.size[0] == len(paths_out)
    #     img_shape = X[0].shape
    img_shape = get_img(data_in[0]).shape
    g = tf.Graph()
    batch_size = batch_size
    curr_num = 0
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
         tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()

        # saver.restore(sess, checkpoint)  # intended only for .ckpt one file checkpoint
        saver.restore(sess, checkpoint)

        num_iters = 1  # int(len(paths_out) / batch_size) ---force 1 iteration---
        for i in range(num_iters):
            pos = i * batch_size
            curr_batch_out = response[pos:pos + batch_size]
            #X = get_img(data_in[0]).shape
            X = np.zeros(batch_shape, dtype=np.float32)
            X[0]=get_img(data_in[0])

            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            for j, response in enumerate(curr_batch_out):
                save_img(response, _preds[j])
                #  save to img_dict using session_key as key
                session_key=request.session.session_key
                img_dict[session_key]=response


def ffwd_to_img(request,img_in, checkpoint, device='/cpu'):
    response=HttpResponse(content_type='img/png')
    #paths_in, paths_out = [in_path], [out_path]
    response=[response]
    img_in=[img_in]
    ffwd(request,img_in, response, checkpoint, batch_size=1, device_t=device)


# def ffwd_different_dimensions(in_path, out_path, checkpoint_dir,
#                               device_t=DEVICE, batch_size=4):
#     in_path_of_shape = defaultdict(list)
#     out_path_of_shape = defaultdict(list)
#     for i in range(len(in_path)):
#         in_image = in_path[i]
#         out_image = out_path[i]
#         shape = "%dx%dx%d" % get_img(in_image).shape
#         in_path_of_shape[shape].append(in_image)
#         out_path_of_shape[shape].append(out_image)
#     for shape in in_path_of_shape:
#         print('Processing images of shape %s' % shape)
#         ffwd(in_path_of_shape[shape], out_path_of_shape[shape],
#              checkpoint_dir, device_t, batch_size)

def evaluate(request,img_in,checkpoint):  # stores an HttpResponse object image in a local dict

    ffwd_to_img(request,img_in, checkpoint,
                device='/cpu')
# def _clean(request):
#     try:
#         session_key=request.session.session_key
#         del img_dict[session_key]
#     except:
#         pass

# def img_return(request,download=None):
#     session_key=request.session.session_key
#     if not session_key:
#         return render(request,'generic/no_active_session.html') # no active session
#     try:
#         response = img_dict[session_key]
#     except:
#         return redirect('/style_transfer/')
#     if not download:
#         return response
#     else:
#         response['Content-Disposition'] = 'attachment; filename="result.png"'
#         return response

