import express from "express";
import Event from "../models/Event.js";

const router = express.Router();

// get all events
router.get("/", async (req, res) => {
  const events = await Event.find().populate("station_id");
  res.json(events);
});

// get event by ID
router.get("/:id", async (req, res) => {
  const event = await Event.findById(req.params.id).populate("station_id");
  res.json(event);
});

// post new event
router.post("/", async (req, res) => {
  const event = new Event(req.body);
  await event.save();
  res.json(event);
});

// put update event
router.put("/:id", async (req, res) => {
  const event = await Event.findByIdAndUpdate(req.params.id, req.body, { new: true });
  res.json(event);
});

// delete event
router.delete("/:id", async (req, res) => {
  await Event.findByIdAndDelete(req.params.id);
  res.json({ message: "Event deleted" });
});

export default router;
