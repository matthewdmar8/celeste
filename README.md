# Initialize CELESTE
Follow the below steps to initialize CELESTE.

            METHOD A - DOCKER
 - Pull down the docker image.
 - Run the docker image and map the ports to the host machine.
 
            METHOD B - VIRTUALENV
 - Clone the repository, create, and activate a new virtualenv.
 - Run [ pip install -r requirements.txt ]
 - Open a python session and run [ import nltk; nltk.download('punkt'); nltk.download('wordnet') ] (This only needs to be done once)
 - Run [ python tars.py] 
 
 HTTP 'Post' requests can now be sent to the IP address of the host machine, from machines on the same wifi network. Use the '/chat' endpoint. (I.E. 172.168.0.1:8000/chat). Be sure to include the API key in the Auth header.
 - You can also use the UI client to send requests. Make sure you've typed your API key and machine's IP address into the side navigation bar's respective slots.
 For the UI, run it by installing vite to your machine. 
                        npm install vite
 
 Then, run the script 
                        npm run dev
