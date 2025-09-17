import express from "express";
import Optimization from "../models/Optimization.js";

const router = express.Router();

// get all optimizations
router.get("/", async (req, res) => {
  const optimizations = await Optimization.find();
  res.json(optimizations);
});

// get optimization by ID
router.get("/:id", async (req, res) => {
  const optimization = await Optimization.findById(req.params.id);
  res.json(optimization);
});

// post new optimization scenario
router.post("/", async (req, res) => {
  const optimization = new Optimization(req.body);
  await optimization.save();
  res.json(optimization);
});

// delete optimization
router.delete("/:id", async (req, res) => {
  await Optimization.findByIdAndDelete(req.params.id);
  res.json({ message: "Optimization scenario deleted" });
});

export default router;
