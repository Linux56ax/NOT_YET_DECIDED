<script>
  import { onMount } from 'svelte';
  import { trains } from '../stores/trains';

  let trainList = [];

  onMount(() => {
    const unsubscribe = trains.subscribe(value => {
      trainList = value;
    });

    return () => {
      unsubscribe();
    };
  });
</script>

<style>
  .train-list {
    list-style-type: none;
    padding: 0;
  }

  .train-item {
    padding: 10px;
    border-bottom: 1px solid #ccc;
  }

  .train-item:last-child {
    border-bottom: none;
  }
</style>

<h2>Live Train Tracking</h2>
<ul class="train-list">
  {#each trainList as train}
    <li class="train-item">
      <strong>{train.name}</strong> - {train.status} - {train.location}
    </li>
  {/each}
</ul>