export interface Train {
    id: string;
    name: string;
    currentStation: string;
    nextStation: string;
    status: 'On Time' | 'Delayed' | 'Cancelled';
    arrivalTime: string;
    departureTime: string;
}

export interface Station {
    code: string;
    name: string;
    location: {
        latitude: number;
        longitude: number;
    };
}