<script lang="ts">
    import Dialog, { Title, Content, Actions } from "@smui/dialog";
    import Button, { Label } from "@smui/button";
    import Textfield from "@smui/textfield";
    import CircularProgress from '@smui/circular-progress';
    import Checkbox from '@smui/checkbox';

    export let open;
    export let cur_user;
    export let all_reports;
    let name = "";
    let email = "";
    // which_reports_to_submit is an array of booleans that tracks whether the report
    // in the corresponding index of all_reports should be submitted to AVID.
    let which_reports_to_submit = [];
    for (let i = 0; i < all_reports.length; i++) {
        which_reports_to_submit.push(false);
    }

    let promise_submit = Promise.resolve(null);
    function handleSubmitReport() {
        promise_submit = submitReport();
    }

    async function submitReport() {
        //Get the relevant reports
        let submitted_reports = [];
        for (let i = 0; i < which_reports_to_submit.length; i++) {
            if (which_reports_to_submit[i]) {
                submitted_reports.push(all_reports[i])
            }
        }

        let req_params = {
            cur_user: cur_user,
            reports: JSON.stringify(submitted_reports),
            name: name,
            email: email,
        };

		let params = new URLSearchParams(req_params).toString();
        const response = await fetch("./submit_avid_report?" + params);
        const text = await response.text();
        const data = JSON.parse(text);
        return data;
    }

</script>

<div>
    <Dialog
        bind:open
        aria-labelledby="simple-title"
        aria-describedby="simple-content"
    >
        <!-- Title cannot contain leading whitespace due to mdc-typography-baseline-top() -->
        <Title id="simple-title">Send All Audit Reports</Title>
        <Content id="simple-content">
            <!-- Description -->
            <div>
                <b>When you are ready to send all of your audit reports to the <a href="https://avidml.org/" target="_blank">AI Vulnerability Database</a> (AVID), please fill out the following information.</b>
                Only your submitted reports will be stored in the database for further analysis. While you can submit reports anonymously, we encourage you to provide your name and/or email so that we can contact you if we have any questions.
            </div>

            <!-- Summary of complete reports -->
            <div>
                <p><b>Summary of Reports Eligible to Send</b> (Reports that include all fields)</p>
                <p>    Select the reports you want to submit. </p>
                <ul>
                    {#each all_reports as report, index}
                        {#if (report["evidence"].length > 0) && (report["text_entry"] != "") && (report["sep_selection"])}

                            <input type="checkbox" bind:checked={which_reports_to_submit[index]} />

                            <span>{report["title"]}</span>
                            <ul>
                                <li>Error Type: {report["error_type"]}</li>
                                <li>Evidence: Includes {report["evidence"].length} example{(report["evidence"].length > 1) ? 's' : ''}</li>
                                <li>Summary/Suggestions: {report["text_entry"]}</li>
                                <li>Audit Category: {report["sep_selection"] || ''}</li>
                            </ul>
                        {/if}
                    {/each}
                </ul>
            </div>

            <!-- Form fields -->
            <div>
                <Textfield bind:value={name} label="(Optional) Name" style="width: 90%" />
            </div>
            <div>
                <Textfield bind:value={email} label="(Optional) Contact email" style="width: 90%" />
            </div>

            <!-- Submission and status message -->
            <div class="dialog_footer">
                <Button on:click={handleSubmitReport} variant="outlined" disabled={which_reports_to_submit.filter(item => item).length == 0}>
                    <Label>Submit Report to AVID</Label>
                </Button>

                <div>
                    <span style="color: grey"><i>
                    {#await promise_submit}
                        <CircularProgress style="height: 32px; width: 32px;" indeterminate />
                    {:then result}
                        {#if result}
                        Successfully sent reports! You may close this window.
                        {/if}
                    {:catch error}
                        <p style="color: red">{error.message}</p>
                    {/await}
                    </i></span>
                </div>
            </div>
        </Content>
    </Dialog>
</div>

<style>
    :global(.mdc-dialog__surface) {
        min-width: 50%;
        min-height: 50%;
        margin-left: 30%;
    }

    .dialog_footer {
        padding: 20px 0px;
    }
</style>
