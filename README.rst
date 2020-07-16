Project DeepSpeech
==================


.. image:: https://readthedocs.org/projects/deepspeech/badge/?version=latest
   :target: http://deepspeech.readthedocs.io/?badge=latest
   :alt: Documentation


.. image:: https://community-tc.services.mozilla.com/api/github/v1/repository/mozilla/DeepSpeech/master/badge.svg
   :target: https://community-tc.services.mozilla.com/api/github/v1/repository/mozilla/DeepSpeech/master/latest
   :alt: Task Status


DeepSpeech is an open source Speech-To-Text engine, using a model trained by machine learning techniques based on `Baidu's Deep Speech research paper <https://arxiv.org/abs/1412.5567>`_. Project DeepSpeech uses Google's `TensorFlow <https://www.tensorflow.org/>`_ to make the implementation easier.

Documentation for installation, usage, and training models are available on `deepspeech.readthedocs.io <http://deepspeech.readthedocs.io/?badge=latest>`_.

For the latest release, including pre-trained models and checkpoints, `see the latest release on GitHub <https://github.com/mozilla/DeepSpeech/releases/latest>`_.

For contribution guidelines, see `CONTRIBUTING.rst <CONTRIBUTING.rst>`_.

For contact and support information, see `SUPPORT.rst <SUPPORT.rst>`_.

Changes and Instructions
########################

We are currently using v0.7.4 of DeepSpeech with a modified scorer which includes vocabulary unique to Carnegie Mellon. All the instructions below assume you are running this code on the TBD_Engine.

Before doing anything you must install the correct dependencies after cloning the repo as follows as follows:

.. code-block:: bash

   cd DeepSpeech
   python3 -m venv venv
   source venv/bin/activate
   pip3 install --upgrade pip==20.0.2 wheel==0.34.2 setuptools==46.1.3
   pip3 install --upgrade -e .
   pip3 uninstall tensorflow
   pip3 install 'tensorflow-gpu==1.15.2'

The scorer we modified was created similar to the original by installing the LibriSpeech normalized LM training as follows:

.. code-block:: bash

   wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
   
We extracted and modified this text file to include phrases including CMU specific vocabulary. This file and all other language model files are located on the TBD_Engine in ``/media/data/deepspeech-training/language-model-files`` except for the scorer (which is located in the repo in ``/data/lm``.
   
We then removed the KenLM source install from ``native_client``, installed it directly from source and built it:

.. code-block:: bash

   cd native_client
   rm -rf kenlm
   git clone https://github.com/kpu/kenlm.git
   mkdir -p build
   cd build && cmake ..
   make j -4
   
Then we built the scorer according to instructions on `the DeepSpeech docs <https://deepspeech.readthedocs.io/en/v0.7.4/Scorer.html>`_. To modify this scorer you need to modify ``librispeech-lm-norm.txt`` located in the ``langage-model-files`` folder on the TBD_Engine and follow the instructions on the above link. Make sure to save the scorer to ``data/lm``.

If you would like to test to scorer on phrases you can run the python file in ``util/kenlm_test``, but first you must install the kenlm dependencies:

.. code-block:: bash
   
   pip install https://github.com/kpu/kenlm/archive/master.zip
   
The uses the binary file stored in ``/media/data/deepspeech-training/language-model-files``.

To train the model from a checkpoint you can either use the original checkpoint from v0.7.4 stored on the TBD_Engine @ ``/media/data/deepspeech-training/deepspeech-0.7.4-checkpoint`` or our current checkpoint @ ``/media/data/deepspeech-training/checkpoints``. Your training data must be in wav format and also have a csv with each row containing wav_filename, wav_filesize, and wav_transcript. An example of how this should look is in ``/media/data/deepspeech-training/training-data``

.. code-block:: bash
   
   export CUDA_VISIBLE_DEVICES=1
   python3 DeepSpeech.py --train_files /media/data/deepspeech-training/<path to train files csv> --n_hidden 2048 --train_cudnn --epochs <number of epochs> --checkpoint_dir <path to checkpoint dir> --learning_rate 0.001
   
Note: When training from the 0.7.4 checkpoint you must specify a different load and save checkpoint director as follows:

.. code-block:: bash
   
   export CUDA_VISIBLE_DEVICES=1
   python3 DeepSpeech.py --train_files /media/data/deepspeech-training/<path to train files csv> --n_hidden 2048 --train_cudnn --epochs <number of epochs> --load_checkpoint_dir <path to load checkpoints> --save_checkpoint_dir <path to save checkpoints --learning_rate 0.001
   
To test your model:

.. code-block:: bash
   
   export CUDA_VISIBLE_DEVICES=1
   python3 DeepSpeech.py --test_files /media/data/deepspeech-training/<path to test files csv> --checkpoint_dir <path to checkpoint>
   
If you have any questions, contact @prithupareek.
   
