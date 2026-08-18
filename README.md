## How to run the scripts with Python
Make the clone of the repository in any directory you desire, and enter the project directory:

`cd /somefolder/myprojects`

`git clone https://github.com/bluecake-dh/scaffold-intersection.git`

`cd scaffold-intersection`

Create a virtual environment named .venv:

`python3 -m venv .venv`

Install the necessary packages:

`pip install python-dotenv numpy opencv-python inference`

You can now run the local model by running `python3 scaffold_analysis_local.py`.
To run the Roboflow-hosted model, copy your Roboflow API key, create a file named .env, and
paste the following line:

`MY_API_KEY=paste your copied API key here`

You can now run the Roboflow-hosted mode by running `python3 scaffold_analysis_api.py`.
