import { writable } from 'svelte/store';

const trains = writable([]);

const updateTrains = (newTrainData) => {
    trains.update(currentTrains => {
        // Update the train data with new information
        const updatedTrains = currentTrains.map(train => {
            const newData = newTrainData.find(t => t.id === train.id);
            return newData ? { ...train, ...newData } : train;
        });
        return updatedTrains.concat(newTrainData.filter(newTrain => 
            !currentTrains.some(train => train.id === newTrain.id)
        ));
    });
};

export { trains, updateTrains };