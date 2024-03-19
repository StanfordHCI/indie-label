# IndieLabel
**End-User Audits: A System Empowering Communities to Lead Large-Scale Investigations of Harmful Algorithmic Behavior**

Michelle S. Lam, Mitchell L. Gordon, Danaë Metaxa, Jeffrey T. Hancock, James A. Landay, Michael S. Bernstein (CSCW 2022)

This repo shares our implementation of **IndieLabel**—an interactive web application for end-user auditing that we introduced in our CSCW paper.

> Because algorithm audits are conducted by technical experts, audits are necessarily limited to the hypotheses that experts think to test. End users hold the promise to expand this purview, as they inhabit spaces and witness algorithmic impacts that auditors do not. In pursuit of this goal, we propose end-user audits—system-scale audits led by non-technical users—and present an approach that scaffolds end users in hypothesis generation, evidence identification, and results communication. Today, performing a system-scale audit requires substantial user effort to label thousands of system outputs, so we introduce a collaborative filtering technique that leverages the algorithmic system's own disaggregated training data to project from a small number of end user labels onto the full test set. Our end-user auditing tool, IndieLabel, employs these projected labels so that users can rapidly explore where their opinions diverge from the algorithmic system's outputs. By highlighting topic areas where the system is under-performing for the user and surfacing sets of likely error cases, the tool guides the user in authoring an audit report. In an evaluation of end-user audits on a popular comment toxicity model with 17 non-technical participants, participants both replicated issues that formal audits had previously identified and also raised previously underreported issues such as under-flagging on veiled forms of hate that perpetuate stigma and over-flagging of slurs that have been reclaimed by marginalized communities.

---

## Installation / Setup
- Activate your virtual environment (tested with Python 3.8).
- Install requirements:
    ```
    $ pip install -r requirements.txt
    ```
- Download and unzip the `data` sub-directory from [this Drive folder](https://drive.google.com/file/d/1iYueqzG9qIB45HT_5iwJp-44Dhfm2XbR/view?usp=sharing) and place it in the repo directory (39.7MB zipped, 124.3MB unzipped).


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
