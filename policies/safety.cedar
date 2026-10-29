// Safety policy for kiln firing schedules, enforced by the agent's `validate` tool.
// Units: degC, degC per hour, minutes, kg, percent. Values below are PLACEHOLDERS: set them from your
// simulator or a ceramics reference, and keep them in sync with src/kiln/config.py
// (tests/test_safety.py fails if the two disagree).
// Evaluated by Cedar (AWS open-source policy language). Anything other than a clean Allow is a rejection.

permit(principal, action == Action::"ApproveSchedule", resource);

@reason("unknown clay body (known: earthenware, stoneware, porcelain)")
forbid(principal, action == Action::"ApproveSchedule", resource)
when { !(["earthenware", "stoneware", "porcelain"].contains(resource.clay_body)) };

@reason("load must be above 0 kg and no more than the kiln capacity of 800 kg")
forbid(principal, action == Action::"ApproveSchedule", resource)
when { resource.load_kg.lessThanOrEqual(decimal("0.0")) || resource.load_kg.greaterThan(decimal("800.0")) };

@reason("soak time must be 10 to 240 minutes")
forbid(principal, action == Action::"ApproveSchedule", resource)
when { resource.soak_minutes.lessThan(decimal("10.0")) || resource.soak_minutes.greaterThan(decimal("240.0")) };

// Quartz inversion at 573 degC: heating through it too fast cracks ware.
@reason("heating up to 573 degC must be 20 to 150 degC/h (quartz inversion)")
forbid(principal, action == Action::"ApproveSchedule", resource)
when { resource.ramp_0_573.lessThan(decimal("20.0")) || resource.ramp_0_573.greaterThan(decimal("150.0")) };

@reason("wet ware (moisture above 2%) must heat up to 573 degC at no more than 75 degC/h")
forbid(principal, action == Action::"ApproveSchedule", resource)
when { resource.moisture_pct.greaterThan(decimal("2.0")) && resource.ramp_0_573.greaterThan(decimal("75.0")) };

@reason("heating from 573 degC to peak must be 30 to 300 degC/h")
forbid(principal, action == Action::"ApproveSchedule", resource)
when { resource.ramp_573_peak.lessThan(decimal("30.0")) || resource.ramp_573_peak.greaterThan(decimal("300.0")) };

@reason("cooling must be 20 to 250 degC/h (faster risks dunting)")
forbid(principal, action == Action::"ApproveSchedule", resource)
when { resource.cooling_rate.lessThan(decimal("20.0")) || resource.cooling_rate.greaterThan(decimal("250.0")) };

@reason("earthenware peak temperature must be 950 to 1120 degC")
forbid(principal, action == Action::"ApproveSchedule", resource)
when { resource.clay_body == "earthenware" && (resource.peak_temp.lessThan(decimal("950.0")) || resource.peak_temp.greaterThan(decimal("1120.0"))) };

@reason("stoneware peak temperature must be 1180 to 1300 degC")
forbid(principal, action == Action::"ApproveSchedule", resource)
when { resource.clay_body == "stoneware" && (resource.peak_temp.lessThan(decimal("1180.0")) || resource.peak_temp.greaterThan(decimal("1300.0"))) };

@reason("porcelain peak temperature must be 1250 to 1400 degC")
forbid(principal, action == Action::"ApproveSchedule", resource)
when { resource.clay_body == "porcelain" && (resource.peak_temp.lessThan(decimal("1250.0")) || resource.peak_temp.greaterThan(decimal("1400.0"))) };

// The model's prediction is an input to the decision too.
@reason("predicted defect rate is above the allowed maximum")
forbid(principal, action == Action::"ApproveSchedule", resource)
when { resource.predicted_defect_pct.greaterThan(context.max_defect_pct) };
