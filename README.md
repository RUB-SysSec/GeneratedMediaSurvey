# $PAPER_TITLE

This repository contains the code for our $TIER_ONE_CONFERENCE paper $PAPER_TITLE.

> $ABSTRACT

## Survey

The code for our survey can be found in the `webapp` directory.
We offer two modes to deploy the application. One for local development, one for testing the deployment in Docker.
You will need to perform step 0 (data setup) regardless of the setup you choose.

### 0. Data

Put your data in `webapp/data` and `webapp/image_data` and do not change the directory structure.
We also distribute some example data.


### 1. Local

You can locally deploy an instance of our webapp.
This is probably the most convinient method for development.
All you need to do is to install the package (in this case in editable mode and with test dependencies):

```cli
cd webapp
pip install -e ".[testing]"
```

Then you can build the database and deploy a development server:

```cli
FLASK_APP=dfsurvey.app.factory FLASK_CONFIG=default flask deploy build 
FLASK_APP=dfsurvey.app.factory FLASK_CONFIG=default flask run
```

For a full list of cli commands see [here](webapp/dfsurvey/app/cli.py) and full list of config options [here](webapp/dfsurvey/config.py).

The tests can be found under `tests`.

```cli
pytest
```

You will need a working version of [Chrome](https://www.google.com/chrome/) and [Selenium](https://www.selenium.dev/documentation/) to run all tests.

### 2. Docker

We also offer a full [dockerized](https://docs.docker.com/get-started/) environment which is used to deploy the survey.
You can choose between three different configurations:

- **Dev**: Which deploys the local installation with a wsgi server (no-persistent database).
- **Stage**: Is an addition to the dev mode for testing the setup with a PostgreDB. 
- **Prod**: Deploys the full setup with an outward facing [NGINX](https://www.nginx.com/) which acts as a reverse-proxy for two backend-end services (one side-by-side; one rating based). This configuration needs further setup (described below).

### Build and deploy

You can build and deploy the survey as follows:

#### Dev

```cli
docker-compose -f docker-compose.dev.yml build
docker-compose -f docker-compose.dev.yml up -d
```

#### Stage

Since stage is just an augment you have to provide both files:

```cli
docker-compose -f docker-compose.dev.yml -f docker-compose.stage.yml build
docker-compose -f docker-compose.dev.yml -f docker-compose.stage.yml up -d
```

#### Prod

```cli
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
```

### 4. Exporting Results

After you have run the survey you can export the results via the following command (substiute your own credentials where necessary):

```cli
FLASK_APP=dfsurvey.app.factory FLASK_CONFIG=export POSTGRES_USER=replaceme POSTGRES_PASSWORD=replaceme POSTGRES_DB=replaceme flask export json
```

### Configure the production environment

The production environment is designed to be run on an actual domain server with a known domain.
Thus, it requires some configuration upfront:

- Fill in your domain name in [nginx/nginx.conf](nginx/nginx.conf) under server name. The setup will automatically generate a valid ssl certificate using [certbot](https://certbot.eff.org/).
- Complete the [prod.env](prod.env) file with your desired configuration.
- Enter your domain and mail adress in [init-letsencrypt.sh](init-letsencrypt.sh).


## Analysis

The code for the statistical analysis can be found in the `$NOTEBOOK_DIRECTORY` directory.
