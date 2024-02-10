Collab link: https://colab.research.google.com/drive/13TfmmVpNc8EpNaUGPI7h3ZJ-b2CLR-dU#scrollTo=RHYix-Vwek-D

Torch Faster R-CNN for Object Detection and Instance Segmentation

Welcome to our comprehensive repository, designed to facilitate the streamlined implementation of Torch Faster R-CNN for object detection and instance segmentation with CUDA acceleration. This framework is crafted to be highly accessible, enabling both training and video processing with minimal setup, and harnesses the combined power of PyTorch, Roboflow, and CUDA for optimal performance.
Overview

This solution aims to democratize advanced computer vision capabilities, allowing users to effortlessly train models and process videos directly from YouTube. It integrates seamlessly with Roboflow for dataset management, optimizing for performance on CUDA-supported hardware, and provides a host of features to enhance the user experience and model effectiveness.
Features

*    One-Line Training: Initiate model training with just a single command, compatible with CLI or Jupyter notebooks.
*    CUDA Integration: Automatic detection and utilization of CUDA for performance optimization on supported hardware.
*    Roboflow Integration: Streamlined dataset loading, including automatic downloads and data preparation with automatic class management.
*    Custom Image Processing: Adapts COCO annotations for compatibility with Faster R-CNN, including bbox format conversion.
*    Model Customization: Facilitates easy adjustment of hyperparameters for tailored training experiences.
*    Video Processing: Processes videos from YouTube, allowing immediate application and evaluation of trained models on real-world data.
*    Intersection Checking: Enables tracking interactions between specified classes in video analysis for enhanced performance insights.
*    Clean-up Option: Option to automatically delete downloaded datasets and videos post-processing to maintain a clean workspace.

- Quick Start

Clone the repository and navigate to the project directory:

bash

    !git clone https://github.com/ghadfield32/torchrcnn_coco_roboflow

    %cd torchrcnn_coco_roboflow/

    !pip install -r requirements.txt

- Roboflow info get:
    1) export roboflow dataset to get:
    !pip install roboflow

    from roboflow import Roboflow
    rf = Roboflow(api_key="api key")
    project = rf.workspace("workspace_name").project("project_name")
    dataset = project.version(11).download("coco-segmentation")

    ^if project_name has -1234 or any numbers after the name, include that in the project_name but not in the project_folder_name

- Train the model with:

bash 

    !python train.py --api_key api_key \
                    --workspace workspace_name \
                    --project_name project_name \
                    --project_folder_name project_name_without_end \
                    --mode train 

    Options to add:
                    --delete_folder_and_video True \
                    --version 11 \
                    --hidden_layer 256 \
                    --lr 0.005 \
                    --num_epochs 1 \
                    --threshold 0.6 \

- To process a video and analyze the results, update the mode and specify the YouTube URL:

bash

    !python train.py --api_key api_key \
        --workspace workspace_name \
        --project_name project_name \
        --project_folder_name project_name_without_end \
        --mode process_video \
        --video_url "<YouTube URL>" \
        --video_name "Youtube video name"

    Options to add:
        --delete_folder_and_video True \
        --check_intersections True \
        --classes_to_track Basketball Hoop \

- Installation

    !git clone https://github.com/ghadfield32/torchrcnn_coco_roboflow

    %cd torchrcnn_coco_roboflow/

    !pip install -r requirements.txt

- Dataset Preparation

    Goto Roboflow, create a repo, and load in pictures to annotate. They have an easy GUI. 

- Training

    !python train.py --api_key api_key \
                    --workspace workspace_name \
                    --project_name project_name \
                    --project_folder_name project_name_without_end \
                    --mode train 

    Options to add:
                    --delete_folder_and_video True \
                    --version 11 \
                    --hidden_layer 256 \
                    --lr 0.005 \
                    --num_epochs 1 \
                    --threshold 0.6 \

- Model Inference and Video Processing

    !python train.py --api_key api_key \
        --workspace workspace_name \
        --project_name project_name \
        --project_folder_name project_name_without_end \
        --mode process_video \
        --video_url "<YouTube URL>" \
        --video_name "Youtube video name"

    Options to add:
        --delete_folder_and_video True \
        --check_intersections True \
        --classes_to_track Basketball Hoop \

- We are open to enhancements, including but not limited to:

    Automatic uploads to Hugging Face for model sharing.
    ByteTracker integration for advanced object tracking.
    Support for larger datasets and exploration of vision transformers.
    Autodistill feature for model optimization.

- Contributions

Your contributions can help take this project to new heights. Whether you're adding features, improving functionality, or expanding the dataset, we welcome your involvement. See the Contributing section for details on how to contribute.

- Acknowledgments

- Special thanks to:

    Hugging Face: For simplifying model sharing and access.
    Roboflow: For streamlining the process of annotating and managing datasets.
    PyTorch Team: For developing the Faster R-CNN architecture and comprehensive guides.
    Daniel Bourke: For the invaluable resource "Learn PyTorch for Deep Learning: Zero to Mastery".


License

This project is open source and available under the MIT License.

Refer to the individual sections for more detailed information on setup, usage, customization, and contributing to this project.