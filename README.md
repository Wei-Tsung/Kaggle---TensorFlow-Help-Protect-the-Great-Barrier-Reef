# Kaggle---TensorFlow-Help-Protect-the-Great-Barrier-Reef

Solution Ranking
--
Silver Medal

Goal of the Competition
--
The goal of this competition is to accurately identify starfish in real-time by building an object detection model trained on underwater videos of coral reefs.


Your work will help researchers identify species that are threatening Australia's Great Barrier Reef and take well-informed action to protect the reef for future generations.

Metric
--


Context
--
Australia's stunningly beautiful Great Barrier Reef is the world’s largest coral reef and home to 1,500 species of fish, 400 species of corals, 130 species of sharks, rays, and a massive variety of other sea life.

Unfortunately, the reef is under threat, in part because of the overpopulation of one particular starfish – the coral-eating crown-of-thorns starfish (or COTS for short). Scientists, tourism operators and reef managers established a large-scale intervention program to control COTS outbreaks to ecologically sustainable levels.

Dataset Description
--

In this competition, you will predict the presence and position of crown-of-thorns starfish in sequences of underwater images taken at various times and locations around the Great Barrier Reef. Predictions take the form of a bounding box together with a confidence score for each identified starfish. An image may contain zero or more starfish.

This competition uses a hidden test set that will be served by an API to ensure you evaluate the images in the same order they were recorded within each video. When your submitted notebook is scored, the actual test data (including a sample submission) will be availabe to your notebook.

Files
train/ - Folder containing training set photos of the form video_{video_id}/{video_frame_number}.jpg.

[train/test].csv - Metadata for the images. As with other test files, most of the test metadata data is only available to your notebook upon submission. Just the first few rows available for download.

video_id - ID number of the video the image was part of. The video ids are not meaningfully ordered.
video_frame - The frame number of the image within the video. Expect to see occasional gaps in the frame number from when the diver surfaced.
sequence - ID of a gap-free subset of a given video. The sequence ids are not meaningfully ordered.
sequence_frame - The frame number within a given sequence.
image_id - ID code for the image, in the format '{video_id}-{video_frame}'
annotations - The bounding boxes of any starfish detections in a string format that can be evaluated directly with Python. Does not use the same format as the predictions you will submit. Not available in test.csv. A bounding box is described by the pixel coordinate (x_min, y_min) of its upper left corner within the image together with its width and height in pixels.
example_sample_submission.csv - A sample submission file in the correct format. The actual sample submission will be provided by the API; this is only provided to illustrate how to properly format predictions. The submission format is further described on the Evaluation page.

example_test.npy - Sample data that will be served by the example API.

greatbarrierreef - The image delivery API that will serve the test set pixel arrays. You may need Python 3.7 and a Linux environment to run the example offline without errors.

Time-series API Details
The API serves the images one by one, in order by video and frame number, as pixel arrays.

Expect to see roughly 13,000 images in the test set.

The API will require roughly two GB of memory after initialization. The initialization step (env.iter_test()) will require meaningfully more memory than that; we recommend you do not load your model until after making that call. The API will also consume less than ten minutes of runtime for loading and serving the data.
