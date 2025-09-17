import express from "express";
import KPI from "../models/KPI.js";

const router = express.Router();

// get all KPIs
router.get("/", async (req, res) => {
  const kpis = await KPI.find();
  res.json(kpis);
});

// get kpi by ID
router.get("/:id", async (req, res) => {
  const kpi = await KPI.findById(req.params.id);
  res.json(kpi);
});

// post new kpi record
router.post("/", async (req, res) => {
  const kpi = new KPI(req.body);
  await kpi.save();
  res.json(kpi);
});

// delete kpi record
router.delete("/:id", async (req, res) => {
  await KPI.findByIdAndDelete(req.params.id);
  res.json({ message: "KPI record deleted" });
});

export default router;
