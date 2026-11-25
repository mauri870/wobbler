"""validation/gost_fetch.py: the parts of the client of ESA's Gaia Observation Forecast Tool (GOST, https://gaia.esac.esa.int/gost/) that do not need the network, on the two replies that GOST gave on 2026-11-24 (tests/golden/gost_submit_reply.html: the reply to a submitted position; tests/golden/gost_result_page.html: the result page of the session after it)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "validation"))
import gost_fetch  # noqa: E402

GOLDEN = Path(__file__).parent / "golden"
SUBMIT = (GOLDEN / "gost_submit_reply.html").read_text()
RESULT = (GOLDEN / "gost_result_page.html").read_text()


def test_the_report_id_of_a_submission_is_read_from_the_reply():
    assert gost_fetch.parse_submitted(SUBMIT) == "1795547033"


def test_a_reply_that_is_an_error_is_refused_with_its_text_and_is_not_retried():
    with pytest.raises(gost_fetch.GostError, match="error occured: null"):
        gost_fetch.parse_submitted("<h4>The following error occured: null</h4>")
    with pytest.raises(gost_fetch.GostError, match="captcha"):
        gost_fetch.parse_submitted("<html>Please solve the captcha</html>")


def test_the_result_page_gives_the_status_and_the_csv_link_of_a_report_by_its_id():
    report = gost_fetch.parse_result(RESULT, "1795547033")
    assert report["status"] == "COMPLETED" and report["csv"] == "export.jsp?id=DA16BD639F2FC06C974E5D0460894827/1795547033&format=csv", report
    assert "0 hits" in report["message"], report


def test_a_report_that_the_page_does_not_list_is_an_error_naming_the_id():
    with pytest.raises(gost_fetch.GostError, match="999"):
        gost_fetch.parse_result(RESULT, "999")


def test_batches_hold_at_most_the_size_and_keep_every_position_once_in_order():
    positions = [(f"s{i}", 10.0 + i, -5.0) for i in range(25)]
    batches = gost_fetch.batches(positions, 10)
    assert [len(b) for b in batches] == [10, 10, 5] and [p for b in batches for p in b] == positions
    assert gost_fetch.batches([], 10) == []
    with pytest.raises(ValueError, match="10000"):
        gost_fetch.batches(positions, 10_001)


def test_the_uploaded_csv_has_the_name_and_the_two_angles_in_degrees_and_a_name_with_a_comma_is_refused():
    assert gost_fetch.positions_csv([("a", 1.5, -2.25), ("b", 359.99, 89.0)]) == "a,1.5,-2.25\nb,359.99,89.0\n"
    with pytest.raises(ValueError, match="comma"):
        gost_fetch.positions_csv([("a,b", 1.0, 2.0)])


def test_the_targets_of_an_export_are_checked_against_the_batch_and_the_ones_without_rows_are_named():
    export = "Target, ra[rad]\nbh3,1\nbh3,2\nnode0,3\n"
    assert gost_fetch.targets_of(export) == {"bh3": 2, "node0": 1}
    assert gost_fetch.check_export(export, ["bh3", "node0", "bh1"]) == ["bh1"]
    with pytest.raises(gost_fetch.GostError, match="stranger"):
        gost_fetch.check_export(export + "stranger,4\n", ["bh3", "node0"])


class FakeOpener:
    """Replays replies in the order of the form's flow: the page, the submission, the result page (once for every entry of `results`), the CSV export."""

    def __init__(self, results, export="Target, ra[rad]\nbh3,1\n"):
        self.results, self.export, self.requests = list(results), export, []

    def open(self, request, timeout=None):
        url = request if isinstance(request, str) else request.full_url
        self.requests.append(url)
        if url.endswith("GostServlet"):
            body = SUBMIT
        elif url.endswith("result.jsp"):
            body = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        elif "export.jsp" in url:
            body = self.export
        else:
            body = "<html>the form</html>"

        class Reply:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *arguments):
                return False

            def read(self_inner):
                return body.encode()

        return Reply()


def test_a_report_that_the_result_page_does_not_list_yet_is_waited_for_and_then_fetched(monkeypatch):
    """The first run of 5,000 positions stopped at 'the result page does not list the report', which a report still being made may be (argued, not known)."""
    monkeypatch.setattr(gost_fetch.time, "sleep", lambda seconds: None)
    empty = "<html><table><tr><td>nothing</td></tr></table></html>"
    opener = FakeOpener([empty, empty, RESULT])
    export = gost_fetch.fetch_batch([("bh3", 1.0, 2.0)], "2014-08-01T00:00:00", "2017-05-29T00:00:00", 0, opener)
    assert export.startswith("Target") and sum(url.endswith("result.jsp") for url in opener.requests) == 3, opener.requests


def test_a_report_never_listed_stops_at_the_cap_with_the_error_and_the_last_page_saved(monkeypatch, tmp_path):
    monkeypatch.setattr(gost_fetch.time, "sleep", lambda seconds: None)
    empty = "<html><table><tr><td>nothing</td></tr></table></html>"
    opener = FakeOpener([empty])
    with pytest.raises(gost_fetch.GostError, match="1795547033.*20"):
        gost_fetch.fetch_batch([("bh3", 1.0, 2.0)], "2014-08-01T00:00:00", "2017-05-29T00:00:00", 0, opener, save_to=tmp_path)
    assert sum(url.endswith("result.jsp") for url in opener.requests) == gost_fetch.POLLS and (tmp_path / "result_page_1795547033.html").read_text() == empty


def test_a_status_that_is_neither_complete_nor_in_progress_still_stops_at_once(monkeypatch):
    monkeypatch.setattr(gost_fetch.time, "sleep", lambda seconds: None)
    failed = RESULT.replace("<td>COMPLETED</td>", "<td>FAILED</td>", 1)
    with pytest.raises(gost_fetch.GostError, match="FAILED"):
        gost_fetch.fetch_batch([("bh3", 1.0, 2.0)], "2014-08-01T00:00:00", "2017-05-29T00:00:00", 0, FakeOpener([failed]))
