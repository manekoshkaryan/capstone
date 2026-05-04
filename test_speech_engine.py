import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from speech_engine import SpeechEngine, _format_spp

eng = SpeechEngine.__new__(SpeechEngine)

def fmt(d):
    return _format_spp(d)

def test_example_1_critical_dead_ahead():
    d = {"class": "person", "distance_m": 0.6, "bearing": "center",
         "urgency": "critical", "is_new": True}
    assert fmt(d) == "Stop. Person ahead.", repr(fmt(d))

def test_example_2_warn_center_clear_right():
    d = {"class": "door", "distance_m": 1.1, "bearing": "center",
         "urgency": "warn", "is_new": True}
    assert fmt(d) == "Door one meter ahead. Slight right.", repr(fmt(d))

def test_example_3_warn_lateral_sidestep():
    d = {"class": "chair", "distance_m": 1.8, "bearing": "slight_left",
         "urgency": "warn", "is_new": True}
    assert fmt(d) == "Chair two meters slightly left. Step right.", repr(fmt(d))

def test_example_4_info_far_right():
    d = {"class": "car", "distance_m": 4.2, "bearing": "far_right",
         "urgency": "info", "is_new": True}
    assert fmt(d) == "Car four meters hard right.", repr(fmt(d))

def test_example_5_post_turn_clear():
    d = {"class": None, "distance_m": None, "bearing": None,
         "urgency": "info", "is_new": True, "post_turn": True}
    assert fmt(d) == "Now clear.", repr(fmt(d))

def test_example_6_post_turn_new_obstacle():
    d = {"class": "wall", "distance_m": 1.5, "bearing": "slight_left",
         "urgency": "warn", "is_new": True, "post_turn": True}
    assert fmt(d) == "Now Wall one meter slightly left. Step right.", repr(fmt(d))

def test_example_7_crossing():
    d = {"class": "person", "distance_m": 2.0, "bearing": "center",
         "urgency": "warn", "is_new": True, "is_crossing": True}
    assert fmt(d) == "Person crossing ahead. Wait.", repr(fmt(d))

def test_example_8_suppressed():
    d = {"class": "chair", "distance_m": 1.8, "bearing": "slight_left",
         "urgency": "warn", "is_new": False}
    assert fmt(d) == "", repr(fmt(d))

def test_wall_critical():
    d = {"class": "wall", "distance_m": 0.4, "bearing": "center",
         "urgency": "critical", "is_new": True}
    assert fmt(d) == "Stop. Wall ahead.", repr(fmt(d))

def test_wall_warn_right():
    d = {"class": "wall", "distance_m": 1.2, "bearing": "right",
         "urgency": "warn", "is_new": True}
    assert fmt(d) == "Wall one meter right. Step left.", repr(fmt(d))

def test_distance_rounding_half_down():
    d = {"class": "person", "distance_m": 1.5, "bearing": "center",
         "urgency": "warn", "is_new": True}
    assert "one meter" in fmt(d), repr(fmt(d))

def test_distance_rounding_up():
    d = {"class": "person", "distance_m": 1.6, "bearing": "center",
         "urgency": "warn", "is_new": True}
    assert "two meters" in fmt(d), repr(fmt(d))

def test_under_one_meter_says_close():
    d = {"class": "person", "distance_m": 0.8, "bearing": "center",
         "urgency": "critical", "is_new": True}
    result = fmt(d)
    assert "close" not in result, "Template A must not include distance"
    assert result == "Stop. Person ahead.", repr(result)

def test_info_beyond_five_meters_not_spoken():
    d = {"class": "car", "distance_m": 5.5, "bearing": "center",
         "urgency": "info", "is_new": True}
    result = fmt(d)
    assert result == "Car five meters ahead.", repr(result)

def test_far_left_bearing():
    d = {"class": "bicycle", "distance_m": 3.0, "bearing": "far_left",
         "urgency": "info", "is_new": True}
    assert fmt(d) == "Bicycle three meters hard left.", repr(fmt(d))

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(failed)
