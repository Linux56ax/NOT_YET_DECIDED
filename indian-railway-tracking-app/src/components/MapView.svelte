<script>
  import { onMount } from 'svelte';
  import { trains } from '../stores/trains';
  import L from 'leaflet';

  let map;
  let markers = [];

  onMount(() => {
    map = L.map('map').setView([20.5937, 78.9629], 5); // Centered on India

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© OpenStreetMap'
    }).addTo(map);

    const unsubscribe = trains.subscribe(trainData => {
      updateMarkers(trainData);
    });

    return () => {
      unsubscribe();
      markers.forEach(marker => map.removeLayer(marker));
    };
  });

  function updateMarkers(trainData) {
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];

    trainData.forEach(train => {
      const marker = L.marker([train.latitude, train.longitude]).addTo(map);
      marker.bindPopup(`<b>${train.name}</b><br>Status: ${train.status}`);
      markers.push(marker);
    });
  }
</script>

<style>
  #map {
    height: 100%;
    width: 100%;
  }
</style>

<div id="map"></div>