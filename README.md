# IndieLabel

## Installation / Setup
- Activate your virtual environment (tested with Python 3.8).
- Install requirements:
    ```
    $ pip install -r requirements.txt
    ```
- Download and unzip the `data` sub-directory from [this Drive folder](https://drive.google.com/file/d/1In9qAzV5t--rMmEH2R5miWpZ4IQStgFu/view?usp=sharing) and place it in the repo directory (334.2MB zipped, 549.1MB unzipped).


- Start the Flask server:
    ```
    $ python server.py
    ```

- Concurrently build and run the Svelte app in another terminal session:
    ```
    $ cd indie_label_svelte/
    $ HOST=0.0.0.0 PORT=5000 npm run dev autobuild
    ```

- You can now visit `localhost:5001` to view the IndieLabel app!

## Main paths
Here's a summary of the relevant pages used for each participant in our study. For easier setup and navigation, we added URL parameters for the different labeling and auditing modes used in the study.
- Participant's page: `localhost:5001/?user=<USER_NAME>`
- Labeling task pages:
    - Group-based model (group selection): `localhost:5001/?user=<USER_NAME>&tab=labeling&label_mode=3`
    - End-user model (data labeling): `localhost:5001/?user=<USER_NAME>&tab=labeling&label_mode=0`
- Tutorial page: `localhost:5001/?user=DemoUser&scaffold=tutorial `
- Auditing task pages:
    - Fixed audit, end-user model: `localhost:5001/?user=<USER_NAME>&scaffold=personal`
    - Fixed audit, group-based model: `localhost:5001/?user=<USER_NAME>&scaffold=personal_group`
    - Free-form audit, end-user model: `localhost:5001/?user=<USER_NAME>&scaffold=prompts` 

## Setting up a new model
- Set up your username and navigate to the **Labeling** page 
    - Option A: Using a direct URL parameter
        - Go to `localhost:5001/?user=<USER_NAME>&tab=labeling&label_mode=0`, where in place of `<USER_NAME>`, you've entered your desired username
    - Option B: Using the UI
        - Go to the Labeling page and ensure that the "Create a new model" mode is selected.
        - Select the User button on the top menu and enter your desired username.

- Label all of the examples in the table
    - When you're done, click the "Get Number of Comments Labeled" button to verify the number of comments that have been labeled. If there are at least 40 comments labeled, the "Train Model" button will be enabled.
    - Click on the "Train Model" button and wait for the model to train (~30-60 seconds).
    
- Then, go to the **Auditing** page and use your new model.
    - To view the different auditing modes that we provided for our evaluation task, please refer to the URL paths listed in the "Auditing task pages" section above.
