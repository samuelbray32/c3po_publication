import datajoint as dj
from spyglass.utils.dj_mixin import SpyglassMixin
from spyglass.common.common_nwbfile import AnalysisNwbfile

schema = dj.schema("sambray_c3po_fig5")

@schema
class BehavTable(SpyglassMixin, dj.Manual):
    definition = """
    nwb_file_name: varchar(255)
    ---
    -> AnalysisNwbfile
    behav_object_id: varchar(40)
    """

@schema
class CompiledStateScriptDataframe(SpyglassMixin, dj.Manual):
    definition = """
    nwb_file_name: varchar(255)
    ---
    -> AnalysisNwbfile
    script_object_id: varchar(40)
    """