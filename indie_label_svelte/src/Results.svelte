<script lang="ts">
    import { onMount } from "svelte";
    import ClusterResults from "./ClusterResults.svelte";

    import Button, { Label } from "@smui/button";
    import LinearProgress from "@smui/linear-progress";
    import Checkbox from '@smui/checkbox';
    import DataTable, {
        Head,
        Body,
        Row,
        Cell,
        Label,
        SortValue,
    } from "@smui/data-table";
    import FormField from "@smui/form-field";

    let cur_examples = [];
    let promise = Promise.resolve(null);

    let scaffold_methods = ["personal", "personal_group", "prompts"];

    let all_users = [];
	async function getUsers() {    
		const response = await fetch("./get_users");
		const text = await response.text();
		const data = JSON.parse(text);
		all_users = data["users"];
        promise = getResults();
	}

    onMount(async () => {
        getUsers()
    });
    
    async function getResults() {
        let req_params = {
            users: all_users
        };
        let params = new URLSearchParams(req_params).toString();
        const response = await fetch("./get_results?" + params);
        const text = await response.text();
        const data = JSON.parse(text);

        let results = data["results"];
        return results;
    }

    function get_complete_ratio(reports) {
        let total = reports.length;
        let complete = reports.filter(item => item.complete_status).length;
        return "" + complete + "/" + total + " complete";
    }

    function get_complete_count(reports) {
        return reports.filter(item => item.complete_status).length;
    }

    function get_summary(reports) {
        let summary = "";
        let total_audits = 0
        for (const scaffold_method of scaffold_methods) {
            if (reports[scaffold_method]) {
                let cur_reports = reports[scaffold_method];
                let cur_ratio = get_complete_ratio(cur_reports);
                let cur_result = "<li><b>" + scaffold_method + "</b>: " + cur_ratio + "</li>";
                summary += cur_result;
                let cur_complete = get_complete_count(cur_reports);
                total_audits += cur_complete;
            }
        }

        let top_summary = "<li><b>Total audits</b>: " + total_audits + "</li>";
        summary = "<ul>" + top_summary + summary + "</ul>";
        return summary;
    }

    function get_url(user, scaffold_method) {
        return "http://localhost:5001/?user=" + user + "&scaffold=" + scaffold_method;
    }
</script>

<svelte:head>
    <title>Results</title>
</svelte:head>

<div class="panel">
    <div class="panel_contents">
        <div>
            <h3>Results</h3>
        </div>

        <div style="padding-top:50px">
            {#await promise}
                <div class="app_loading">
                    <LinearProgress indeterminate />
                </div>
            {:then results}
                {#if results}
                    {#each results as user_report}
                        <div class="head_3">{user_report["user"]}</div>
                        <div class="section_indent">
                            <div class="head_5">Summary</div>
                            <div>{@html get_summary(user_report)}</div>
                            <ul>
                                <li>Labeling pages
                                    <ul>
                                        <li>
                                            <a href="http://localhost:5001/?user={user_report["user"]}&tab=labeling&label_mode=3" target="_blank">Group-based model</a>
                                        </li>
                                        <li>
                                            <a href="http://localhost:5001/?user={user_report["user"]}&tab=labeling&label_mode=0" target="_blank">Personalized model</a>
                                        </li>
                                    </ul>
                                </li>
                                <li>Auditing pages
                                    <ul>
                                        <li>
                                            <a href="http://localhost:5001/?user={user_report["user"]}&scaffold=personal_group" target="_blank">Group-based audit - personal scaffold</a>
                                        </li>
                                        <li>
                                            <a href="http://localhost:5001/?user={user_report["user"]}&scaffold=personal" target="_blank">Individual audit - personal scaffold</a>
                                        </li>
                                        <li>
                                            <a href="http://localhost:5001/?user={user_report["user"]}&scaffold=prompts" target="_blank">Individual audit - prompt scaffold</a>
                                        </li>
                                    </ul>
                                </li>
                            </ul>
                        </div>
                        {#each scaffold_methods as scaffold_method}
                            {#if user_report[scaffold_method]}
                            <div class="spacing_vert_60 section_indent">
                                <div class="head_5">
                                    {scaffold_method} ({get_complete_ratio(user_report[scaffold_method])})
                                    [<a href={get_url(user_report["user"], scaffold_method)} target="_blank">link</a>]
                                </div>
                                {#each user_report[scaffold_method] as report}
                                    <div class="spacing_vert_40 section_indent">
                                        <div class="head_6_non_cap">
                                            {report["title"]}
                                        </div>

                                        <div class="spacing_vert_20">
                                            <div class="">
                                                <b>Error type</b>
                                            </div>
                                            {report["error_type"]}
                                        </div>

                                        <div class="spacing_vert_20">
                                            <div class="">
                                                <b>Evidence</b>
                                            </div>
                                            {#if report["evidence"].length > 0}
                                            <ClusterResults 
                                                cluster={null} 
                                                model={null} 
                                                data={{"cluster_comments": report["evidence"]}} 
                                                show_vis={false} 
                                                show_checkboxes={false} 
                                                table_width_pct={100} 
                                                rowsPerPage={10} 
                                                table_id={"panel"}
                                            />
                                            {:else}
                                                <p class="grey_text">
                                                    No examples added
                                                </p>
                                            {/if}
                                        </div>

                                        <div class="spacing_vert_20">
                                            <div class="">
                                                <b>Summary/Suggestions</b>
                                            </div>
                                            {report["text_entry"]}
                                        </div>

                                        <div class="spacing_vert_20">
                                            <b>Completed</b>
                                            <FormField>
                                                <Checkbox checked={report["complete_status"]} disabled/>
                                            </FormField>
                                        </div>
                                        
                                    </div>
                                {/each}
                            </div>
                            {/if}
                        {/each}
                    {/each}
                {/if}
            {:catch error}
                <p style="color: red">{error.message}</p>
            {/await}
        </div>
    </div>
</div>

<style>
    .panel {
        width: 80%;
        padding: 50px;
    }
</style>
