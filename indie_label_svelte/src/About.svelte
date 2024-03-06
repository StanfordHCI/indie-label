<script lang="ts">
</script>

<svelte:head>
    <title>About</title>
</svelte:head>

<div class="panel">
    <div class="panel_contents">
        <div>
            <h3>About IndieLabel</h3>
            <p class="body">
                The IndieLabel deployment on this Hugging Face Space is a collaboration between researchers at the <a href="https://hci.stanford.edu/" target="_blank">Stanford HCI Group</a> and <a href="https://avidml.org/arva/" target="_blank">ARVA (AI Risk and Vulnerability Alliance)</a>. The IndieLabel tool is a research prototype designed to empower everyday users to lead large-scale investigations of harmful algorithmic behavior, and this Space is intended as a public-facing deployment of this prototype.
            </p>

            <div class="card-container">
                <p class="head_5">IndieLabel & the End-User Audits paper</p>
                <p class="body">
                    The <a href="https://github.com/StanfordHCI/indie-label" target="_blank">IndieLabel</a> tool was initially presented as a research prototype system in an academic publication called <b><i>End-User Audits: A System Empowering Communities to Lead Large-Scale Investigations of Harmful Algorithmic Behavior</i></b>, presented at CSCW 2022. The full paper is available on the <a href="https://dl.acm.org/doi/10.1145/3555625" target="_blank">ACM Digital Library</a> or via <a href="https://hci.stanford.edu/publications/2022/Lam_EndUserAudits_CSCW22.pdf" target="_blank">Stanford HCI</a>.
                </p>
                <p class="body">
                    The key question underlying the End-User Audits paper is: <i>how do we enable everyday users to lead large-scale algorithm audits</i>? IndieLabel aids users in rapidly identifying where they disagree with a system's behavior with the help of a personalized recommender system model.
                </p>

                <p class="body">
                    Our paper abstract summarizes our core motivation, approach, and findings:
                    "Because algorithm audits are conducted by technical experts, audits are necessarily limited to the hypotheses that experts think to test. End users hold the promise to expand this purview, as they inhabit spaces and witness algorithmic impacts that auditors do not. In pursuit of this goal, we propose end-user audits—system-scale audits led by non-technical users—and present an approach that scaffolds end users in hypothesis generation, evidence identification, and results communication. Today, performing a system-scale audit requires substantial user effort to label thousands of system outputs, so we introduce a collaborative filtering technique that leverages the algorithmic system's own disaggregated training data to project from a small number of end user labels onto the full test set. Our end-user auditing tool, IndieLabel, employs these predicted labels so that users can rapidly explore where their opinions diverge from the algorithmic system's outputs. By highlighting topic areas where the system is under-performing for the user and surfacing sets of likely error cases, the tool guides the user in authoring an audit report. In an evaluation of end-user audits on a popular comment toxicity model with 17 non-technical participants, participants both replicated issues that formal audits had previously identified and also raised previously underreported issues such as under-flagging on veiled forms of hate that perpetuate stigma and over-flagging of slurs that have been reclaimed by marginalized communities."
                </p>

                <p class="body">
                    The End-User Audits work was led by Stanford PhD student <a href="http://michelle123lam.github.io/" target="_blank">Michelle Lam</a> along with co-authors <a href="https://mgordon.me/" target="_blank">Mitchell Gordon</a>, <a href="https://metaxa.net/" target="_blank">Danaë Metaxa</a>, <a href="https://sml.stanford.edu/people/jeff-hancock" target="_blank">Jeffrey Hancock</a>, <a href="https://www.landay.org/" target="_blank">James Landay</a>, and <a href="https://hci.stanford.edu/msb/" target="_blank">Michael Bernstein</a>.
                </p>

                <div class="image-container">
                    <div>
                        <img src="./img/lam.jpg" alt="Michelle Lam">
                        <p><b>Michelle Lam</b><br/>PhD Student, Stanford CS</p>
                    </div>
                    <div>
                        <img src="./img/gordon.jpeg" alt="Mitchell Gordon">
                        <p><b>Mitchell Gordon</b><br/>Incoming Asst Professor, MIT EECS</p>
                    </div>
                    <div>
                        <img src="./img/metaxa.jpg" alt="Danaë Metaxa">
                        <p><b>Danaë Metaxa</b><br/>Asst Professor, UPenn CIS</p>
                    </div>
                    <div>
                        <img src="./img/hancock.png" alt="Jeffrey Hancock">
                        <p><b>Jeffrey Hancock</b><br/>Professor, Stanford Comm</p>
                    </div>
                    <div>
                        <img src="./img/landay.jpeg" alt="James Landay">
                        <p><b>James Landay</b><br/>Professor, Stanford CS</p>
                    </div>
                    <div>
                        <img src="./img/bernstein.jpeg" alt="Michael Bernstein">
                        <p><b>Michael Bernstein</b><br/>Assoc Professor, Stanford CS</p>
                    </div>
                </div>
            </div>

            <div class="card-container">
                <p class="head_5">ARVA</p>
                <p class="body">
                    To fill in: information about ARVA and AVID.
                </p>
            </div>

            <div class="card-container">
                <p class="head_5">The Team</p>
                <p class="body">
                    The IndieLabel Hugging Face deployment was made possible by a wonderful team of volunteers who worked on adapting IndieLabel for use by a general audience, connecting its reports to AVID (AI Vulnerability Database), and deploying it on Hugging Face Spaces. The team includes:
                </p>
                <div>
                    <div class="image-container">
                        <div>
                            <img src="./img/lam.jpg" alt="Michelle Lam">
                            <p><b>Michelle Lam</b></p>
                        </div>
                        <div>
                            <img src="./img/temp.png" alt="Carol Anderson">
                            <p><b>Carol Anderson</b></p>
                        </div>
                        <div>
                            <img src="./img/temp.png" alt="Christina Pan">
                            <p><b>Christina Pan</b></p>
                        </div>
                        <div>
                            <img src="./img/temp.png" alt="Nathan Butters">
                            <p><b>Nathan Butters</b></p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<style>
    a {
        color: #333;
        text-decoration: underline;
    }

    .head_5 {
        margin: 10px 0;
    }

    .card-container {
        margin: 40px 0;
    }

    .image-container {
        margin: 20px 0;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 20px; /* Adjust the gap between images */
    }
    
    .image-container img {
        max-width: 110px; /* Adjust the maximum width of images */
        height: 110px;
        border-radius: 50%;
        display: block;
    }

    .image-container p {
        text-align: center;
        margin-top: 5px; /* Adjust the margin between image and name */
        max-width: 110px;
        font-size: 11px;
    }
</style>
