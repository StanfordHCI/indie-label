<script lang="ts">
    import Card, { Content } from '@smui/card';
</script>

<svelte:head>
    <title>About</title>
</svelte:head>

<div class="panel">
    <div class="panel_contents">
        <div>
            <h3>IndieLabel Tutorial</h3>
            <p class="body">
                Check out this 5-minute video tutorial for a quick overview of the IndieLabel tool. If you prefer a written document, you can also find a help article <a href="https://docs.google.com/document/d/1BTo8LFUriiZcSWCcQy0lel9VxGl4KZtlFnu7wAFaiu4/edit?usp=sharing" target="_blank">here</a>.
            </p>
            <div class="video-container">
                <iframe src="https://www.youtube.com/embed/Je0DCDnJ6KQ?si=H1dVy-oe6PSP0QYM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
            </div>
            <div class="info-card">
                <Card variant="outlined" style="background-color: rgb(242 236 255); padding: 10px 20px;">
                    <p><b>Note</b>: IndieLabel is a research prototype. It is primarily designed for educational purposes to demonstrate our end-user auditing method applied towards a real-world system.</p>
                </Card>
            </div>
        </div>
        <div>
            <h3>About IndieLabel</h3>
            <p class="body">
                The IndieLabel deployment on this Hugging Face Space is a collaboration between researchers at the <a href="https://hci.stanford.edu/" target="_blank">Stanford HCI Group</a> and <a href="https://avidml.org/arva/" target="_blank">ARVA (AI Risk and Vulnerability Alliance)</a>. The IndieLabel tool is a research prototype designed to empower everyday users to lead large-scale investigations of harmful algorithmic behavior, and this Space is intended as a public-facing deployment of this prototype.
            </p>

            <div class="card-container">
                <p class="head_5">IndieLabel & the End-User Audits paper</p>
                <p class="body">
                    The <a href="https://github.com/StanfordHCI/indie-label" target="_blank">IndieLabel</a> tool was initially presented as a research prototype in an academic publication called <b><i>End-User Audits: A System Empowering Communities to Lead Large-Scale Investigations of Harmful Algorithmic Behavior</i></b>. IndieLabel aids users in rapidly identifying where they disagree with a system's behavior with the help of a personalized recommender system model.
                </p>
                <p class="body">
                    The key question underlying the End-User Audits paper is: <i>how do we enable everyday users to lead large-scale algorithm audits</i>? The full paper was presented at CSCW 2022 and is available on the <a href="https://dl.acm.org/doi/10.1145/3555625" target="_blank">ACM Digital Library</a> or via <a href="https://hci.stanford.edu/publications/2022/Lam_EndUserAudits_CSCW22.pdf" target="_blank">Stanford HCI</a>. The End-User Audits paper abstract summarizes our core motivation, approach, and findings:
                </p>
                <p class="body blockquote">
                    "Because algorithm audits are conducted by technical experts, audits are necessarily limited to the hypotheses that experts think to test. End users hold the promise to expand this purview, as they inhabit spaces and witness algorithmic impacts that auditors do not. In pursuit of this goal, we propose end-user audits—system-scale audits led by non-technical users—and present an approach that scaffolds end users in hypothesis generation, evidence identification, and results communication. Today, performing a system-scale audit requires substantial user effort to label thousands of system outputs, so we introduce a collaborative filtering technique that leverages the algorithmic system's own disaggregated training data to project from a small number of end user labels onto the full test set. Our end-user auditing tool, IndieLabel, employs these predicted labels so that users can rapidly explore where their opinions diverge from the algorithmic system's outputs. By highlighting topic areas where the system is under-performing for the user and surfacing sets of likely error cases, the tool guides the user in authoring an audit report. In an evaluation of end-user audits on a popular comment toxicity model with 17 non-technical participants, participants both replicated issues that formal audits had previously identified and also raised previously underreported issues such as under-flagging on veiled forms of hate that perpetuate stigma and over-flagging of slurs that have been reclaimed by marginalized communities."
                </p>
                

                <p class="body">
                    The End-User Audits work was led by Stanford PhD student Michelle Lam along with co-authors Mitchell Gordon, Danaë Metaxa, Jeffrey Hancock, James Landay, and Michael Bernstein.
                </p>

                <div class="image-container">
                    <div>
                        <img src="./img/lam.jpg" alt="Michelle Lam">
                        <p>
                            <a href="http://michelle123lam.github.io" target="_blank"><b>Michelle Lam</b></a><br>
                            PhD Candidate, Stanford CS
                        </p>
                    </div>
                    <div>
                        <img src="./img/gordon.jpeg" alt="Mitchell Gordon">
                        <p>
                            <a href="https://mgordon.me/" target="_blank"><b>Mitchell Gordon</b></a><br/>
                            Incoming Asst Professor, MIT EECS
                        </p>
                    </div>
                    <div>
                        <img src="./img/metaxa.jpg" alt="Danaë Metaxa">
                        <p>
                            <a href="https://metaxa.net/" target="_blank"><b>Danaë Metaxa</b></a><br/>
                            Asst Professor, UPenn CIS
                        </p>
                    </div>
                    <div>
                        <img src="./img/hancock.png" alt="Jeffrey Hancock">
                        <p>
                            <a href="https://sml.stanford.edu/people/jeff-hancock" target="_blank"><b>Jeffrey Hancock</b></a><br/>
                            Professor, Stanford Comm
                        </p>
                    </div>
                    <div>
                        <img src="./img/landay.jpeg" alt="James Landay">
                        <p>
                            <a href="https://www.landay.org/" target="_blank"><b>James Landay</b></a><br/>
                                Professor, Stanford CS
                            </p>
                    </div>
                    <div>
                        <img src="./img/bernstein.jpeg" alt="Michael Bernstein">
                        <p>
                            <a href="https://hci.stanford.edu/msb/" target="_blank"><b>Michael Bernstein</b></a><br/>
                            Assoc Professor, Stanford CS
                        </p>
                    </div>
                </div>
            </div>

            <div class="card-container">
                <p class="head_5">ARVA</p>
                <p class="body">
                    The <a href="https://avidml.org/arva/" target="_blank">AI Risk and Vulnerability Alliance (ARVA)</a> is a nonprofit organization focused on making AI safer for everyone. Our mission is to empower communities to recognize, diagnose, and manage vulnerabilities in AI that affects them. Our flagship project is the <a href="https://avidml.org" target="_blank">AI Vulnerability Database (AVID)</a>, is an open-source knowledge base of failure modes for AI models, datasets, and systems. 
                </p>
            </div>

            <div class="card-container">
                <p class="head_5">The Team</p>
                <p class="body">
                    The IndieLabel Hugging Face deployment was made possible by a wonderful team of volunteers who worked on adapting IndieLabel for use by a general audience, connecting its reports to AVID (AI Vulnerability Database), and deploying it on Hugging Face Spaces. The team includes:
                </p>
                <div>
                    <div class="image-container-wide">
                        <div class="image-wide">
                            <img src="./img/lam.jpg" alt="Michelle Lam">
                            <p>
                                <a href="http://michelle123lam.github.io" target="_blank"><b>Michelle Lam</b></a><br>
                                Michelle Lam is a PhD Candidate at Stanford University in the HCI Group. Her research focuses on building systems that empower everyday users to surface their expertise to design and evaluate AI systems.
                            </p>
                        </div>
                        <div class="image-wide">
                            <img src="./img/anderson.jpeg" alt="Carol Anderson">
                            <p>
                                <a href="https://www.carol-anderson.com/" target="_blank"><b>Carol Anderson</b></a><br>
                                Carol Anderson is a data scientist and machine learning practitioner with expertise in natural language processing (NLP), biological data, and AI ethics. She serves as AVID’s machine learning lead.
                            </p>
                        </div>
                        <div class="image-wide">
                            <img src="./img/pan.jpg" alt="Christina A. Pan">
                            <p>
                                <a href="https://www.christinaapan.com/" target="_blank"><b>Christina A. Pan</b></a><br>
                                Christina A. Pan started her career building machine learning (ML) models at Google, which inspired her passion for design thinking and AI ethics.
                            </p>
                        </div>
                        <div class="image-wide">
                            <img src="./img/butters.jpeg" alt="Nathan Butters">
                            <p>
                                <b>Nathan Butters</b><br>
                                Nathan Butters is a product manager in the Office of Ethical and Humane Use at Salesforce. He is a cofounder of the AI Risk and Vulnerability Alliance (ARVA).
                            </p>
                        </div>
                        <div class="image-wide">
                            <img src="./img/blili_hamelin.jpeg" alt="Borhane Blili-Hamelin">
                            <p>
                                <a href="https://borhane.xyz/" target="_blank"><b>Borhane Blili-Hamelin</b></a><br>
                                Borhane Blili-Hamelin is an ethicist, researcher and AI risk management consultant. He is an officer at the AI Risk and Vulnerability Alliance (ARVA), an affiliate at Data & Society, and a senior consultant at BABL AI. 
                            </p>
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

    .panel {
        margin: 0 40px;
    }

    .blockquote {
        margin: 0 40px;
    }

    .head_5 {
        margin: 10px 0;
    }

    .card-container {
        margin: 40px 0;
    }

    .info-card {
        margin: 10px 0;
    }

    .info-card p {
        line-height: normal;
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

    .image-container-wide {
        margin: 20px 0;
        display: flex;
        flex-direction: column;
        justify-content: center;
        gap: 20px;
    }

    .image-wide {
        display: flex;
        flex-direction: row;
        align-items: center;
        gap: 10px;
    }
    
    .image-container-wide img {
        max-width: 110px; /* Adjust the maximum width of images */
        height: 110px;
        border-radius: 50%;
        display: block;
    }

    .video-container { 
        position: relative; 
        padding-bottom: 10px; 
        padding-top: 10px; 
        height: 40vh; 
        overflow: hidden; 
    }

    .video-container iframe { 
        position: absolute; 
        top: 0; 
        left: 0; 
        width: 100%; 
        height: 100%; 
    }
</style>
