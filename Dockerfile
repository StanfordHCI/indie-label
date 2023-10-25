FROM node:19

WORKDIR workdir/

COPY indie_label_svelte ./indie_label_svelte/
COPY *.py .
COPY requirements.txt .
COPY *.sh .

RUN apt-get update \
	&& apt install -y python3-pip \
	&& pip3 install -r requirements.txt

RUN npm install --global rollup \
	&& npm install --save-dev svelte rollup-plugin-svelte --legacy-peer-deps \
	&& cd indie_label_svelte \
	&& npm run build

ENV HOST=0.0.0.0
EXPOSE 5001

ENTRYPOINT gdown 1In9qAzV5t--rMmEH2R5miWpZ4IQStgFu \
	&& unzip data.zip \
	&& rm data.zip \
	&& bash run_app.sh & sleep infinity;
















