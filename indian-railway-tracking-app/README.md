# Indian Railway Tracking App

This project is a live tracking application for Indian railway trains built using Svelte and JavaScript. It provides real-time updates on train locations, statuses, and other relevant information.

## Features

- Live tracking of Indian railway trains
- Display of train status and information
- Visualization of train locations on a map
- User-friendly interface for tracking multiple trains

## Project Structure

```
indian-railway-tracking-app
├── src
│   ├── App.svelte          # Main application component
│   ├── main.js             # Entry point for the Svelte application
│   ├── components          # Contains reusable components
│   │   ├── TrainTracker.svelte  # Component for fetching and displaying train data
│   │   ├── TrainList.svelte     # Component for displaying a list of trains
│   │   └── MapView.svelte        # Component for visualizing train locations on a map
│   ├── stores              # Contains Svelte stores for state management
│   │   └── trains.js       # Store for managing train data
│   └── types               # Type definitions for TypeScript
│       └── index.js        # Type definitions for Train and Station
├── public
│   └── index.html          # Main HTML file for the application
├── package.json            # npm configuration file
├── README.md               # Project documentation
└── svelte.config.js        # Svelte configuration settings
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd indian-railway-tracking-app
   ```
3. Install the dependencies:
   ```
   npm install
   ```

## Usage

To start the application, run the following command:
```
npm run dev
```
This will start a development server, and you can view the application in your browser at `http://localhost:5000`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you would like to add.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.