# YouCook2-BoundingBox

We introduce in our BMVC 2018 [paper](http://bmvc2018.org/contents/papers/0070.pdf) the **YouCook2-BoundingBox** dataset which contains spatial bounding boxes for YouCook2 video segments.

We provide bounding box annotations on YouCook2 validation & testing sets for the 67 most frequent objects (see class_file.csv). All videos are resized to have 720px in width while maintaining the aspect ratio.

The following files are included as part of the YouCook2-BoundingBox dataset:
* yc2_bb_skeleton.txt: Describes the expected structure/format of the JSON annotation files.
* yc2_bb_val_annotations.json: Contains the spatial bounding box annotations for videos from our validation split.
* yc2_bb_public_test_annotations.json: Contains only the video dimensions from our testing split. We retain the testing split for server-side evaluation.
* yc2_training_vid.json: Contains only the video dimensions from the training split. Only the testing and validation splits have bounding box annotations, training is only given the sentence annotations as supervision.
* class_file.csv: Contains the complete list of the objects from our class list. Both the singular and plural forms are included.

The [download link](http://youcook2.eecs.umich.edu/download) to YouCook2-BoundingBox.
