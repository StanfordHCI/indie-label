<script lang="ts">
    import Dialog, { Title, Content, Actions } from "@smui/dialog";
    import Button, { Label } from "@smui/button";
    import Textfield from "@smui/textfield";
    import Select, { Option } from "@smui/select";
    import CircularProgress from '@smui/circular-progress';

    export let open;
    export let cur_user;
    export let all_reports;
    let email = "";
    let all_sep_options = [
        "Accuracy",
        "Bias/Discrimination",
        "Adversarial Example",
        "Other",
    ];
    let sep_selection = "";

    let promise_submit = Promise.resolve(null);
    function handleSubmitReport() {
        promise_submit = submitReport();
    }

    async function submitReport() {
        let req_params = {
            cur_user: cur_user,
            reports: JSON.stringify(all_reports),
            email: email,
            sep_selection: sep_selection,
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
                Only your submitted reports will be stored in the database for further analysis. While you can submit reports anonymously, we encourage you to provide your email so that we can contact you if we have any questions.
            </div>

            <!-- Summary of complete reports -->
            <div>
                <p><b>Summary of Reports to Send</b> (Reports that include evidence and are marked as complete)</p>
                <ul>
                    {#each all_reports as report}
                        {#if report["complete_status"] && (report["evidence"].length > 0)}
                            <li>{report["title"]}</li>
                            <ul>
                                <li>Error Type: {report["error_type"]}</li>
                                <li>Evidence: Includes {report["evidence"].length} example{(report["evidence"].length > 1) ? 's' : ''}</li>
                                <li>Summary/Suggestions: {report["text_entry"]}</li>
                            </ul>
                        {/if}
                    {/each}
                </ul>
            </div>

            <!-- Form fields -->
            <div>
                <Select bind:value={sep_selection} label="Audit category" style="width: 90%">
                    {#each all_sep_options as opt}
                        <Option value={opt}>{opt}</Option>
                    {/each}
                </Select>
            </div>
            <div>
                <Textfield bind:value={email} label="(Optional) Contact email" style="width: 90%" />
            </div>

            <!-- Submission and status message -->
            <div class="dialog_footer">
                <Button on:click={handleSubmitReport} variant="outlined">
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
