# Probing_scene_level_representations_in_vision_and_language_models

My research intern at Utrecht University from March 2022 to September 2022, supervised by Dr.Albert Gatt and Michele Cafagna. This Github repository contains my report and code. 

Key takeaways of this project:

• Used an image caption dataset to test the ability of VL models to align scene captions with images, and found single stream
VL models outperform dual stream VL models, mainly due to differences in the visual part of their initial embeddings.

• Compared the performance of VL models to align scene captions and captions in the Microsoft COCO Caption dataset with
images. The accuracy of aligning MS COCO captions is higher than scene captions for all VL models, possibly due to the
higher stylistic similarity between MS COCO captions and the captions used for pre-training VL models compared to scene
captions.

• Conducted an experiment to know which kind of caption is preferred by VL models to describe images. MS COCO captions
are preferred by VL models to describe the image than using scene captions to describe images.

• Proposed a probing task on each layer of VisualBERT which performs the best in aligning scene captions with images, to
know whether there is scene information encoded in its embedding. Found that deeper layers contain more scene information
than the lower layers in VisualBERT.
