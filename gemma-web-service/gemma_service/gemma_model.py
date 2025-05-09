# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import re
from dotenv import load_dotenv

# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras_hub

# set Gemma model version
gemma_model_id = "gemma3_instruct_4b"

def initialize_model():
    """Loads environment variables and configures the Gemma model."""
    load_dotenv()
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    if not kaggle_username:
        raise ValueError("KAGGLE_USERNAME environment variable not found. Did you set it in your .env file?")
    kaggle_key = os.getenv('KAGGLE_KEY')
    if not kaggle_key:
        raise ValueError("KAGGLE_KEY environment variable not found. Did you set it in your .env file?")

    # create instance of Gemma model
    gemma = keras_hub.models.GemmaCausalLM.from_preset(gemma_model_id)
    #gemma.summary() # FOR TESTING ONLY
    return gemma  # Return the initialized model

def create_model_instance():
    """Creates a message processor function with a persistent model."""
"gemma_service/gemma_model.py" 79L, 2676B                                                                                               43,21         26%
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ ./run_service.sh
2025-05-09 18:28:59.280288: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1746815339.303065   10940 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1746815339.310560   10940 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1746815339.328490   10940 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815339.328524   10940 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815339.328527   10940 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815339.328531   10940 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
Traceback (most recent call last):
  File "/home/ardyadipta/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/gemma_service/gemma_service_main.py", line 24, in <module>
    gemma_model = create_model_instance() # initialize model
  File "/home/ardyadipta/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/gemma_service/gemma_model.py", line 49, in create_model_instance
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import re
from dotenv import load_dotenv

# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras_hub

# set Gemma model version
gemma_model_id = "gemma3_instruct_4b"

def initialize_model():
    """Loads environment variables and configures the Gemma model."""
    load_dotenv()
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    if not kaggle_username:
        raise ValueError("KAGGLE_USERNAME environment variable not found. Did you set it in your .env file?")
    kaggle_key = os.getenv('KAGGLE_KEY')
    if not kaggle_key:
        raise ValueError("KAGGLE_KEY environment variable not found. Did you set it in your .env file?")

    # create instance of Gemma model
    gemma = keras_hub.models.Gemma333CausalLM.from_preset(gemma_model_id)
    #gemma.summary() # FOR TESTING ONLY
    return gemma  # Return the initialized model

def create_model_instance():
    """Creates a message processor function with a persistent model."""
"gemma_service/gemma_model.py" 79L, 2679B                                                                                               43,37         26%
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    model = initialize_model()
  File "/home/ardyadipta/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/gemma_service/gemma_model.py", line 43, in initialize_model
    gemma = keras_hub.models.GemmaCausalLM.from_preset(gemma_model_id)
  File "/home/ardyadipta/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/venv/lib/python3.10/site-packages/keras_hub/src/models/task.py", line 192, in from_preset
    cls = find_subclass(preset, cls, backbone_cls)
  File "/home/ardyadipta/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/venv/lib/python3.10/site-packages/keras_hub/src/utils/preset_utils.py", line 105, in find_subclass
    raise ValueError(
ValueError: Unable to find a subclass of GemmaCausalLM that is compatible with Gemma3Backbone found in preset 'gemma3_instruct_4b'.
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ vim gemma_service/gemma_model.py
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ ./run_service.sh
2025-05-09 18:29:36.888405: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1746815376.910792   11032 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1746815376.918284   11032 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1746815376.935982   11032 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815376.936020   11032 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815376.936024   11032 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815376.936027   11032 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
Traceback (most recent call last):
  File "/home/ardyadipta/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/gemma_service/gemma_service_main.py", line 24, in <module>
    gemma_model = create_model_instance() # initialize model
  File "/home/ardyadipta/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/gemma_service/gemma_model.py", line 49, in create_model_instance
    model = initialize_model()
  File "/home/ardyadipta/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/gemma_service/gemma_model.py", line 43, in initialize_model
    gemma = keras_hub.models.Gemma333CausalLM.from_preset(gemma_model_id)
AttributeError: module 'keras_hub.api.models' has no attribute 'Gemma333CausalLM'. Did you mean: 'Gemma3CausalLM'?
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ vim gemma_service/gemma_model.py
#!/bin/bash

# activate virtual environment
source venv/bin/activate

cd gemma_service/
# to allow more than localhost access, add "--host 0.0.0.0":
#uvicorn gemma_service_main:app --host 0.0.0.0 --reload

python3 gemma_service_main.py
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
"run_service.sh" 10L, 236B                                                                                                              1,1           All
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ ./run_service.sh
2025-05-09 18:29:55.890139: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from gemma_model import create_model_instance
from gemma_model import get_model_id

app = FastAPI()
gemma_model = create_model_instance() # initialize model

class Request(BaseModel):
    text: str

class Response(BaseModel):
    text: str

@app.post("/gemma_request/")
async def process_text(request: Request):
    """
    Processes the input text and returns a modified version.
    """
    response_text = gemma_model(request.text)
"gemma_service/gemma_service_main.py" 51L, 1401B                                                                                        1,1           Top
#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
E0000 00:00:1746815395.912685   11058 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has alr
#!/bin/bash

# activate virtual environment
source venv/bin/activate

cd gemma_service/
# to allow more than localhost access, add "--host 0.0.0.0":
#uvicorn gemma_service_main:app --host 0.0.0.0 --reload

python3 gemma_service_main.py
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
"run_service.sh" 10L, 236B                                                                                                              1,1           All
eady been registered
#!/bin/bash

# activate virtual environment
source venv/bin/activate

cd gemma_service/
# to allow more than localhost access, add "--host 0.0.0.0":
#uvicorn gemma_service_main:app --host 0.0.0.0 --reload

python3 gemma_service_main.py
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
~
"run_service.sh" 10L, 236B                                                                                                              10,1          All
E0000 00:00:1746815395.920170   11058 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1746815395.937473   11058 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815395.937503   11058 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815395.937508   11058 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815395.937513   11058 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
Downloading from https://www.kaggle.com/api/v1/models/keras/gemma3/keras/gemma3_instruct_4b/1/download/task.json...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.55k/5.55k [00:00<00:00, 17.8MB/s]
^[[B^[[B^[[B^[[A^[[B./run_service.sh: line 10: 11058 Killed                  python3 gemma_service_main.py
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ ./run_service.sh
2025-05-09 18:31:21.344485: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1746815481.543405   13879 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1746815481.600125   13879 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1746815482.056667   13879 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815482.056716   13879 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815482.056722   13879 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815482.056732   13879 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
./run_service.sh: line 10: 13879 Killed                  python3 gemma_service_main.py
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ vim run_service.sh
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ vim
.env              gemma_service/    requirements.txt  setup_python.sh   venv/
README.md         installation.sh   run_service.sh    tests/
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ vim gemma_service/gemma_service_main.py
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ vim run_service.sh
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ vim run_service.sh
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ ./test/test_post.sh
-bash: ./test/test_post.sh: No such file or directory
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ ls
README.md  gemma_service  installation.sh  requirements.txt  run_service.sh  setup_python.sh  tests  venv
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ cd tests/
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/tests$ ls
test_generation.sh  test_post.sh
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/tests$ chmod +x test_post.sh test_generation.sh
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/tests$ ls
test_generation.sh  test_post.sh
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/tests$ ll
-bash: ll: command not found
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/tests$ cd ..
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ ./tests/test_post.sh
curl: (7) Failed to connect to localhost port 8000: Connection refused

(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ ./run_service.sh
2025-05-09 18:36:33.888261: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1746815794.073305   16478 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1746815794.128042   16478 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1746815794.550661   16478 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815794.550708   16478 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815794.550712   16478 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815794.550715   16478 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
^T./run_service.sh: line 10: 16478 Killed                  python3 gemma_service_main.py
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ ./run_service.sh
2025-05-09 18:37:41.905276: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1746815862.088059   16638 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1746815862.140010   16638 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1746815862.570329   16638 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815862.570378   16638 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815862.570382   16638 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746815862.570386   16638 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
./run_service.sh: line 10: 16638 Killed                  python3 gemma_service_main.py
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ ./run_service.sh
2025-05-09 18:40:12.978961: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1746816013.162252   16971 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1746816013.212270   16971 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1746816013.649799   16971 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746816013.649845   16971 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746816013.649851   16971 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746816013.649861   16971 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
./run_service.sh: line 10: 16971 Killed                  python3 gemma_service_main.py
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ ./run_service.sh
2025-05-09 18:41:16.114568: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1746816076.301815   17129 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1746816076.355776   17129 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1746816076.778911   17129 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746816076.778960   17129 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746816076.778965   17129 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746816076.778970   17129 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
./run_service.sh: line 10: 17129 Killed                  python3 gemma_service_main.py
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ ./run_service.sh
2025-05-09 18:42:44.124507: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1746816164.304553   18284 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1746816164.360387   18284 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1746816164.796144   18284 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746816164.796190   18284 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746816164.796194   18284 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1746816164.796200   18284 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
./run_service.sh: line 10: 18284 Killed                  python3 gemma_service_main.py
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ ls
README.md  gemma_service  installation.sh  requirements.txt  run_service.sh  setup_python.sh  tests  venv
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service$ cd gemma_service/
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/gemma_service$ ls
__pycache__  gemma_model.py  gemma_service_main.py
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/gemma_service$ nvidia-smi
-bash: nvidia-smi: command not found
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/gemma_service$ brew install nvidia-smi
-bash: brew: command not found
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/gemma_service$ lspci | grep -i nvidia
00:03.0 3D controller: NVIDIA Corporation Device 27b8 (rev a1)
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/gemma_service$ sudo apt-get update
sudo apt-get install -y build-essential dkms
Hit:1 https://deb.debian.org/debian bullseye InRelease
Hit:2 https://deb.debian.org/debian-security bullseye-security InRelease
Hit:3 https://deb.debian.org/debian bullseye-updates InRelease
Hit:4 https://deb.debian.org/debian bullseye-backports InRelease
Hit:5 https://download.docker.com/linux/debian bullseye InRelease
Hit:6 https://packages.cloud.google.com/apt gcsfuse-bullseye InRelease
Hit:7 https://deb.nodesource.com/node_22.x nodistro InRelease
Hit:8 https://packages.cloud.google.com/apt google-compute-engine-bullseye-stable InRelease
Hit:9 https://packages.cloud.google.com/apt cloud-sdk-bullseye InRelease
Hit:10 https://packages.cloud.google.com/apt google-cloud-ops-agent-bullseye-2 InRelease
Reading package lists... Done
Reading package lists... Done
Building dependency tree... Done
Reading state information... Done
build-essential is already the newest version (12.9).
dkms is already the newest version (2.8.4-3).
0 upgraded, 0 newly installed, 0 to remove and 11 not upgraded.
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/gemma_service$ nvidia-smi
-bash: nvidia-smi: command not found
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/gemma_service$ lspci | grep -i nvidia
00:03.0 3D controller: NVIDIA Corporation Device 27b8 (rev a1)
(venv) ardyadipta@gemma3-web-service:~/src/gemma-cookbook/Demos/personal-code-assistant/gemma-web-service/gemma_service$ sudo apt-get install -y linux-headers-$(uname -r)
Reading package lists... Done
#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from gemma_model import create_model_instance
from gemma_model import get_model_id

app = FastAPI()
gemma_model = create_model_instance() # initialize model

class Request(BaseModel):
    text: str

class Response(BaseModel):
    text: str

@app.post("/gemma_request/")
async def process_text(request: Request):
    """
    Processes the input text and returns a modified version.
    """
    response_text = gemma_model(request.text)
    response = Response(text=response_text)
    return response

@app.get("/")
async def root():
    return "Gemma server: OK"

@app.get("/info")
async def info():
    return "Gemma service is using: " + get_model_id()

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
~
~
~
"gemma_service/gemma_service_main.py" 51L, 1401B                                                                                                             51,1          All
Building dependency tree... Done
Reading state information... Done
The following additional packages will be installed:
#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import re
from dotenv import load_dotenv

# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras_hub

# set Gemma model version
gemma_model_id = "gemma3_instruct_4b"

def initialize_model():
    """Loads environment variables and configures the Gemma model."""
    load_dotenv()
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    if not kaggle_username:
        raise ValueError("KAGGLE_USERNAME environment variable not found. Did you set it in your .env file?")
    kaggle_key = os.getenv('KAGGLE_KEY')
    if not kaggle_key:
        raise ValueError("KAGGLE_KEY environment variable not found. Did you set it in your .env file?")

    # create instance of Gemma model
    gemma = keras_hub.models.Gemma3CausalLM.from_preset(gemma_model_id)
    #gemma.summary() # FOR TESTING ONLY
    return gemma  # Return the initialized model

def create_model_instance():
    """Creates a message processor function with a persistent model."""
    model = initialize_model()

    def process_message(prompt_text):
        """Processes a message using a local Gemma model."""
        input = f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
        response = model.generate(input, max_length=1024)
"gemma_service/gemma_model.py" 79L, 2677B                                                                                                                    43,35         Top
#
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import re
from dotenv import load_dotenv

# Set the backbend before importing Keras
os.environ["KERAS_BACKEND"] = "jax"
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import keras_nlp
import keras_hub

# set Gemma model version
gemma_model_id = "gemma3_instruct_4b"

def initialize_model():
    """Loads environment variables and configures the Gemma model."""
    load_dotenv()
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    if not kaggle_username:
        raise ValueError("KAGGLE_USERNAME environment variable not found. Did you set it in your .env file?")
    kaggle_key = os.getenv('KAGGLE_KEY')
    if not kaggle_key:
        raise ValueError("KAGGLE_KEY environment variable not found. Did you set it in your .env file?")

    # create instance of Gemma model
    gemma = keras_hub.models.Gemma3CausalLM.from_preset(gemma_model_id)
    #gemma.summary() # FOR TESTING ONLY
    return gemma  # Return the initialized model

def create_model_instance():
    """Creates a message processor function with a persistent model."""
    model = initialize_model()

    def process_message(prompt_text):
        """Processes a message using a local Gemma model."""
        input = f"<start_of_turn>user\n{prompt_text}<end_of_turn>\n<start_of_turn>model\n"
        response = model.generate(input, max_length=1024)
"gemma_service/gemma_model.py" 79L, 2677B                                                                                                                    43,35         Top