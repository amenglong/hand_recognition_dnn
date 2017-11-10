# hand_recognition_dnn
Hand-signs image recognition that recognize numbers from 0 to 5 in sign language.
<ul>
<li>Training set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).</li>
<li>Test set: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).</li>
</ul>

<p align="center"><img src="https://user-images.githubusercontent.com/24521991/32612515-6c0dee0e-c5a3-11e7-82e7-1d872ffd022e.png" width="500"></p>

We implement a <b>3 layers deep neural network</b> using TensorFlow. Notice images resolution is lowered to 64 x 64 pixels before putting into the network. 
<p align="center"><img src="https://user-images.githubusercontent.com/24521991/32612759-0a432efe-c5a4-11e7-8d77-917f73f09cbf.png" width="500"></p>

### hand_dnn_model.py
<ul>
<li>Train the model</li>
<li>Record parameters in a pickle file</li>
</ul>

### hand_dnn_reco.py
<ul>
<li>Load trained model parameters from pickle file </li>
<li>Run chosen image into the network providing prediction </li>
</ul>

### tf_utils.py
<ul>
<li>Contain supporting fuctions for hand_dnn_model.py and hand_dnn_reco.py</li>
</ul>
