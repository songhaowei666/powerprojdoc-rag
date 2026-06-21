import sys
from pathlib import Path

# 将项目根目录加入 sys.path，解决直接运行时的模块导入问题
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import os
import time
import zipfile

import requests

from src.config import settings

api_key = os.getenv("MINERU_API_KEY") or settings.mineru_api_key

_BATCH_UPLOAD_URL = "https://mineru.net/api/v4/file-urls/batch"
_POLL_INTERVAL_SEC = 5


def _auth_headers() -> dict[str, str]:
    """构造 MinerU API 鉴权请求头。"""
    if not api_key:
        raise ValueError("未配置 MINERU_API_KEY，请在 .env 中设置")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _parse_api_response(res: requests.Response) -> dict:
    """解析 MinerU API 响应，非成功时抛出 RuntimeError。"""
    try:
        body = res.json()
    except ValueError as exc:
        raise RuntimeError(f"MinerU API 响应非 JSON，HTTP {res.status_code}") from exc
    if res.status_code != 200 or body.get("code") != 0:
        msg = body.get("msg", res.text)
        raise RuntimeError(f"MinerU API 请求失败: HTTP {res.status_code}, {msg}")
    return body["data"]


def get_task_id(file_name):
    url='https://mineru.net/api/v4/extract/task'
    header = {
        'Content-Type':'application/json',
        "Authorization":f"Bearer {api_key}"
    }
    pdf_url = 'https://vl-image.oss-cn-shanghai.aliyuncs.com/pdf/' + file_name
    data = {
        'url':pdf_url,
        'is_ocr':True,
        'enable_formula': False,
    }

    res = requests.post(url,headers=header,json=data)
    print(res.status_code)
    print(res.json())
    print(res.json()["data"])
    task_id = res.json()["data"]['task_id']
    return task_id

def get_result(task_id):
    url = f'https://mineru.net/api/v4/extract/task/{task_id}'
    header = {
        'Content-Type':'application/json',
        "Authorization":f"Bearer {api_key}"
    }

    while True:
        res = requests.get(url, headers=header)
        result = res.json()["data"]
        print(result)
        state = result.get('state')
        err_msg = result.get('err_msg', '')
        # 如果任务还在进行中，等待后重试
        if state in ['pending', 'running']:
            print("任务未完成，等待5秒后重试...")
            time.sleep(5)
            continue
        # 如果有错误，输出错误信息
        if err_msg:
            print(f"任务出错: {err_msg}")
            return
        # 如果任务完成，下载文件
        if state == 'done':
            full_zip_url = result.get('full_zip_url')
            if full_zip_url:
                local_filename = f"{task_id}.zip"
                print(f"开始下载: {full_zip_url}")
                r = requests.get(full_zip_url, stream=True)
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"下载完成，已保存到: {local_filename}")
                # 下载完成后自动解压
                unzip_file(local_filename)
            else:
                print("未找到 full_zip_url，无法下载。")
            return
        # 其他未知状态
        print(f"未知状态: {state}")
        return


def upload_local_file(
    file_path: str,
    *,
    is_ocr: bool = True,
    enable_formula: bool = False,
    enable_table: bool = True,
    model_version: str = "vlm",
    data_id: str | None = None,
) -> str:
    """
    上传本地 PDF 到 MinerU 并自动提交解析任务。

    :param file_path: 本地 PDF 文件路径
    :return: batch_id，可用于 get_batch_result 轮询结果
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    file_name = os.path.basename(file_path)
    file_entry: dict = {"name": file_name, "is_ocr": is_ocr}
    if data_id is not None:
        file_entry["data_id"] = data_id

    payload = {
        "files": [file_entry],
        "model_version": model_version,
        "enable_formula": enable_formula,
        "enable_table": enable_table,
    }

    res = requests.post(_BATCH_UPLOAD_URL, headers=_auth_headers(), json=payload)
    data = _parse_api_response(res)
    batch_id = data["batch_id"]
    upload_urls = data["file_urls"]

    print(f"batch_id: {batch_id}")
    print(f"开始上传: {file_path}")

    with open(file_path, "rb") as f:
        # 预签名 URL 上传不能设置 Content-Type，否则 OSS 签名会失效
        upload_res = requests.put(upload_urls[0], data=f)
    if upload_res.status_code != 200:
        raise RuntimeError(
            f"文件上传失败: HTTP {upload_res.status_code}, {upload_res.text}"
        )

    print(f"上传完成: {file_name}")
    return batch_id


def get_batch_result(batch_id: str, *, file_index: int = 0) -> None:
    """
    轮询批量解析任务结果，完成后下载并解压 zip。

    :param batch_id: upload_local_file 返回的 batch_id
    :param file_index: extract_result 列表中的文件索引，单文件上传时为 0
    """
    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"

    while True:
        res = requests.get(url, headers=_auth_headers())
        data = _parse_api_response(res)
        extract_results = data.get("extract_result", [])
        if not extract_results:
            print("未找到解析结果，等待后重试...")
            time.sleep(_POLL_INTERVAL_SEC)
            continue

        if file_index >= len(extract_results):
            raise RuntimeError(
                f"file_index={file_index} 超出范围，共 {len(extract_results)} 个文件"
            )

        result = extract_results[file_index]
        print(result)
        state = result.get("state")
        err_msg = result.get("err_msg", "")

        if state in ("waiting-file", "pending", "running", "converting"):
            print(f"任务未完成（{state}），等待{_POLL_INTERVAL_SEC}秒后重试...")
            time.sleep(_POLL_INTERVAL_SEC)
            continue

        if state == "failed" or err_msg:
            print(f"任务出错: {err_msg or state}")
            return

        if state == "done":
            full_zip_url = result.get("full_zip_url")
            if full_zip_url:
                local_filename = f"{batch_id}.zip"
                print(f"开始下载: {full_zip_url}")
                r = requests.get(full_zip_url, stream=True)
                with open(local_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"下载完成，已保存到: {local_filename}")
                unzip_file(local_filename)
            else:
                print("未找到 full_zip_url，无法下载。")
            return

        print(f"未知状态: {state}")
        return


# 解压zip文件的函数
def unzip_file(zip_path, extract_dir=None):
    """
    解压指定的zip文件到目标文件夹。
    :param zip_path: zip文件路径
    :param extract_dir: 解压目标文件夹，默认为zip同名目录
    """
    if extract_dir is None:
        extract_dir = zip_path.rstrip('.zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"已解压到: {extract_dir}")

if __name__ == "__main__":
    # 远程 URL 方式（PDF 需已在 OSS 上）
    # file_name = '【财报】中芯国际：中芯国际2024年年度报告.pdf'
    # task_id = get_task_id(file_name)
    # print('task_id:', task_id)
    # get_result(task_id)

    # 本地上传方式
    local_pdf = "data/test_pdf/巴菲特1964年致合伙人的信年度.pdf"
    batch_id = upload_local_file(local_pdf)
    print("batch_id:", batch_id)
    get_batch_result(batch_id)
