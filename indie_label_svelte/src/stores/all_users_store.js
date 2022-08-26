import { writable } from 'svelte/store';

// Fallback if request doesn't work
let all_users = ["DemoUser"];

export const users = writable(all_users);
