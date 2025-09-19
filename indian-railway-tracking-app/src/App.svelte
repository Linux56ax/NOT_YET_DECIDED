<script>
  import TrainTracker from './components/TrainTracker.svelte';
  import TrainList from './components/TrainList.svelte';
  import MapView from './components/MapView.svelte';

  let trains = [];
</script>

<main>
  <h1>Indian Railway Live Tracking</h1>
  <TrainTracker bind:trains />
  <TrainList {trains} />
  <MapView {trains} />
</main>

<style>
  main {
    text-align: center;
    padding: 1em;
    max-width: 600px;
    margin: 0 auto;
  }

  h1 {
    color: #333;
  }
</style>