import express from "express";
import Station from "../models/Station.js";

const router = express.Router();

// GET all stations
router.get("/", async (req, res) => {
  const stations = await Station.find();
  res.json(stations);
});

// POST new station
router.post("/", async (req, res) => {
  const station = new Station(req.body);
  await station.save();
  res.json(station);
});

export default router;
