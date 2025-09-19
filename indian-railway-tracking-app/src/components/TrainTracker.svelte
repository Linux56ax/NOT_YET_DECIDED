<script>
  import { onMount } from 'svelte';
  import { trains } from '../stores/trains';

  let trainData = [];

  onMount(async () => {
    const fetchTrainData = async () => {
      try {
        const response = await fetch('https://api.railwayapi.com/v2/live/train/{train_number}/date/{date}/apikey/{apikey}/');
        const data = await response.json();
        trainData = data.train;
        trains.set(trainData);
      } catch (error) {
        console.error('Error fetching train data:', error);
      }
    };

    fetchTrainData();
    const interval = setInterval(fetchTrainData, 60000); // Update every minute

    return () => clearInterval(interval); // Cleanup on component destroy
  });
</script>

<style>
  .train-tracker {
    padding: 20px;
    background-color: #f9f9f9;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  .train-info {
    margin-bottom: 10px;
  }
</style>

<div class="train-tracker">
  <h2>Live Train Tracker</h2>
  {#each trainData as train}
    <div class="train-info">
      <strong>Train Number:</strong> {train.number}<br>
      <strong>Status:</strong> {train.status}<br>
      <strong>Current Station:</strong> {train.current_station.name}<br>
      <strong>Arrival Time:</strong> {train.arrival_time}<br>
      <strong>Departure Time:</strong> {train.departure_time}<br>
    </div>
  {/each}
</div>