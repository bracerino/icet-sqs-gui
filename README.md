# SQS-GUI
GUI for generating SQS structures using ICET package. Try the application online at https://sqs-icet.streamlit.app/ or compile it locally (see the manual below). 
Currently implemented only supercell-specific option. 

### **Prerequisities**: 
- Python 3.x (Tested 3.12)
- Console (For Windows, I recommend to use WSL2 (Windows Subsystem for Linux))
- Git (optional for downloading the code)


### **Compile the app**  
Open your terminal console and write the following commands (the bold text):  
(Optional) Install Git:  
      **sudo apt update**  
      **sudo apt install git**    
      
1) Download the XRDlicious code from GitHub (or download it manually without Git on the following link by clicking on 'Code' and 'Download ZIP', then extract the ZIP. With Git, it is automatically extracted):  
      **git clone https://github.com/bracerino/SQS-GUI.git**

2) Navigate to the downloaded project folder:  
      **cd SQS-GUI/**

3) Create a Python virtual environment to prevent possible conflicts between packages:  
      **python3 -m venv sqs_env**

4) Activate the Python virtual environment (before activating, make sure you are inside the SQS-GUI folder):  
      **source sqs_env/bin/activate**
   
5) Install all the necessary Python packages:  
      **pip install -r requirements.txt**

6) Run the SQS-GUI app (always before running it, make sure to activate its Python virtual environment (Step 4):  
      **streamlit run app.py**
