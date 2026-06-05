import logging
from typing import List, Dict, Literal, Union
from pydantic import BaseModel, Field
from src.api_requests import APIProcessor

_log = logging.getLogger(__name__)

# 27家省公司编码映射（占位符，后续可替换为真实编码）
PROVINCE_CODE_BASE = {
    "北京": "001",
    "天津": "002",
    "河北": "003",
    "山西": "004",
    "山东": "005",
    "上海": "006",
    "江苏": "007",
    "浙江": "008",
    "安徽": "009",
    "福建": "010",
    "湖北": "011",
    "湖南": "012",
    "河南": "013",
    "江西": "014",
    "四川": "015",
    "重庆": "016",
    "辽宁": "017",
    "吉林": "018",
    "黑龙江": "019",
    "陕西": "020",
    "甘肃": "021",
    "青海": "022",
    "宁夏": "023",
    "新疆": "024",
    "西藏": "025",
    "内蒙古": "026",
    "广西": "027",
}


# ===== 数据模型 =====

class MetadataField(BaseModel):
    """元数据字段定义"""
    name: str = Field(description="字段名")
    description: str = Field(description="字段说明")
    type: Literal["string", "integer", "float"] = Field(description="字段类型")


class ProvinceCompanyCode(BaseModel):
    """单条省公司编码记录"""
    name: str = Field(description="省公司简称，如'北京'")
    code: str = Field(description="编码占位符，如'001'")


class FilterCondition(BaseModel):
    """单个过滤条件"""
    field: str = Field(description="字段名")
    operator: str = Field(description="操作符，如$eq, $ne, $gt等")
    value: Union[str, int, float, List] = Field(description="过滤值")


class MetadataFilterResponse(BaseModel):
    """元数据过滤条件 LLM 响应"""
    filters: List[FilterCondition] = Field(description="过滤条件列表")
    reasoning: str = Field(description="判断理由")


class AngleItem(BaseModel):
    """单个角度查询（LLM 输出用）"""
    angle_name: str = Field(description="角度名称")
    query_text: str = Field(description="查询文本")
    rationale: str = Field(description="生成理由")


class MultiAngleResponse(BaseModel):
    """多角度查询 LLM 响应"""
    angles: List[AngleItem] = Field(description="角度查询列表")


class QueryAngle(BaseModel):
    """单个角度查询（对外输出）"""
    angle_name: str = Field(description="角度名称，如'semantic_expansion'")
    query_text: str = Field(description="该角度下的查询文本")
    rationale: str = Field(description="生成理由简述")


class PreRetrievalResult(BaseModel):
    """检索前处理统一输出"""
    original_query: str = Field(description="原始查询")
    metadata_filter: Dict = Field(description="ChromaDB where 条件")
    angles: List[QueryAngle] = Field(description="三个角度的查询变体")


# ===== 预定义元数据字段 =====

METADATA_FIELDS = [
    MetadataField(
        name="unit_code",
        description="省公司编码，27家省公司之一，可选值：001~027",
        type="string",
    ),
]


# ===== 工具函数 =====

def _build_province_list_text() -> str:
    """构建省公司列表文本，供 Prompt 使用"""
    lines = []
    for name, code in PROVINCE_CODE_BASE.items():
        lines.append(f"  {code}: {name}")
    return "\n".join(lines)


def _filters_to_where(filters: List[FilterCondition]) -> Dict:
    """将 FilterCondition 列表转换为 ChromaDB where 格式"""
    if not filters:
        return {}
    if len(filters) == 1:
        f = filters[0]
        return {f.field: {f.operator: f.value}}
    # 多个条件用 $and 连接
    conditions = []
    for f in filters:
        conditions.append({f.field: {f.operator: f.value}})
    return {"$and": conditions}


# ===== 处理器类 =====

class MetadataFilterBuilder:
    """元数据过滤条件生成器：根据查询生成 ChromaDB where 条件"""

    def __init__(self, provider: str = "openai"):
        self.processor = APIProcessor(provider=provider)
        self.system_content = (
            "你是元数据过滤条件生成助手。请根据用户的查询，分析需要过滤的元数据字段。\n\n"
            "当前可用元数据字段：\n"
            "- unit_code（string）：省公司编码，可选值：001~027，对应省公司列表如下：\n"
            f"{_build_province_list_text()}\n\n"
            "请仅从上述字段中选择，生成过滤条件。每个条件包含 field、operator、value。\n"
            "支持的 operator：$eq, $ne, $gt, $gte, $lt, $lte, $in, $nin。\n\n"
            "若查询中无法确定任何过滤条件，返回空列表。"
        )

    def build(self, query: str, unit_code: str = None) -> Dict:
        """根据查询生成元数据过滤条件。

        参数：
            query: 用户原始查询
            unit_code: 若已提供，直接组装为 {"unit_code": {"$eq": unit_code}}，不调用 LLM

        返回：
            ChromaDB where 条件字典
        """
        # 短路：已传编码，直接复用
        if unit_code is not None:
            _log.info(f"Unit code provided, skip LLM inference: {unit_code}")
            return {"unit_code": {"$eq": unit_code}}

        # 调用 LLM 推断
        human_content = f"用户查询：{query}"
        result = self.processor.send_message(
            system_content=self.system_content,
            human_content=human_content,
            is_structured=True,
            response_format=MetadataFilterResponse
        )

        if not isinstance(result, dict):
            _log.warning(f"Unexpected result type from LLM: {type(result)}")
            return {}

        filters_data = result.get("filters", [])
        filters = [FilterCondition(**f) for f in filters_data if isinstance(f, dict)]
        where_clause = _filters_to_where(filters)
        _log.info(f"Generated metadata filter: {where_clause}")
        return where_clause


class MultiAngleQueryBuilder:
    """多角度查询构建器：基于原始查询生成 3 个不同角度的检索变体"""

    def __init__(self, provider: str = "openai"):
        self.processor = APIProcessor(provider=provider)
        self.system_content = (
            "你是查询扩展专家。请基于用户的原始查询，从以下三个角度生成检索友好的查询变体：\n\n"
            "1. semantic_expansion：语义扩展，使用同义词、近义表达改写原查询。\n"
            "2. keyword_focus：关键词聚焦，提取核心实体、指标、时间，去掉冗余修饰。\n"
            "3. structured_condition：结构化条件，补充隐含的时间范围、对比维度、限定词。\n\n"
            "每个角度输出查询文本和生成理由。必须严格返回三个角度。"
        )

    def build(self, query: str) -> List[QueryAngle]:
        """构建三个不同角度的查询变体。

        参数：
            query: 用户原始查询

        返回：
            List[QueryAngle]，长度固定为 3
        """
        human_content = f"用户查询：{query}"
        result = self.processor.send_message(
            system_content=self.system_content,
            human_content=human_content,
            is_structured=True,
            response_format=MultiAngleResponse
        )

        if not isinstance(result, dict):
            _log.warning(f"Unexpected result type from LLM: {type(result)}")
            return []

        angles_data = result.get("angles", [])
        angles = [QueryAngle(**a) for a in angles_data if isinstance(a, dict)]
        
        # 确保顺序一致
        angle_order = ["semantic_expansion", "keyword_focus", "structured_condition"]
        angle_map = {a.angle_name: a for a in angles}
        ordered_angles = []
        for name in angle_order:
            if name in angle_map:
                ordered_angles.append(angle_map[name])
            else:
                _log.warning(f"Missing angle in LLM response: {name}")
        
        return ordered_angles


# 默认实例，供其他模块直接导入使用
metadata_filter_builder = MetadataFilterBuilder()
multi_angle_query_builder = MultiAngleQueryBuilder()


# 示例用法
if __name__ == "__main__":
    # 示例一：未传编码，LLM 推断
    query = "北京公司2024年营收增长原因"
    print("=== 示例一：未传编码 ===")
    mf = metadata_filter_builder.build(query)
    print("Metadata filter:", mf)
    
    angles = multi_angle_query_builder.build(query)
    print("Angles:", [a.model_dump() for a in angles])

    # 示例二：已传编码，直接复用
    print("\n=== 示例二：已传编码 ===")
    mf = metadata_filter_builder.build(query, unit_code="001")
    print("Metadata filter:", mf)
