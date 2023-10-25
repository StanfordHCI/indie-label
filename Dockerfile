FROM node:19

# Change node user to 1001 to free up uid 1000
RUN groupmod -g 1001 node \
  && usermod -u 1001 -g 1001 node

# Set up a new user named "user" with user ID 1000 per HF instructions
RUN useradd -m -u 1000 user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app
RUN chmod 777 $HOME/app

COPY --chown=user indie_label_svelte ./indie_label_svelte/
COPY --chown=user *.py .
COPY --chown=user *.sh .
COPY requirements.txt .

RUN apt-get update \
	&& apt install -y python3-pip \
	&& pip3 install -r requirements.txt

WORKDIR $HOME/app/indie_label_svelte
RUN npm install --global rollup \
	&& npm install --save-dev svelte rollup-plugin-svelte --legacy-peer-deps \
	&& npm run build

WORKDIR $HOME/app/
USER user
ENV HOST=0.0.0.0
EXPOSE 5001

RUN gdown 1In9qAzV5t--rMmEH2R5miWpZ4IQStgFu \
	&& unzip data.zip \
	&& rm data.zip

ENTRYPOINT bash run_app.sh
















