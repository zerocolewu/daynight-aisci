"""
用于聚合物 p-smiles 的简单处理函数。

约定：
- p-smiles 使用两个 dummy 原子 `*` 表示聚合位点（且每个 `*` 仅有 1 个邻居）。
- “环化”定义为：将两个 `*` 的邻居原子首尾连接成键，然后删除两个 `*` 原子。
"""

from __future__ import annotations

from typing import Optional, Tuple

from rdkit import Chem


def cyclize_p_smiles(
    p_smiles: str,
    *,
    keep_bond_type: bool = True,
    sanitize: bool = True,
    return_mol: bool = False,
) -> str | Tuple[str, Chem.Mol]:
    """
    将带两个聚合位点 `*` 的 p-smiles 做“环化”（首尾连接）。

    参数：
    - p_smiles: 含两个 `*` 的 SMILES（例如：`*CCCCC*` 或更复杂的重复单元表示）
    - keep_bond_type: True 时使用 `*`-邻居 的键型作为闭环键型；False 则用单键
    - sanitize: 是否对结果分子做 RDKit sanitize（建议开启）
    - return_mol: True 时返回 (smiles, mol)

    返回：
    - 环化后的 canonical smiles（以及可选的 mol）
    """
    mol = Chem.MolFromSmiles(p_smiles)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {p_smiles}")

    star_atoms = [a for a in mol.GetAtoms() if a.GetSymbol() == "*"]
    if len(star_atoms) != 2:
        raise ValueError(f"期望恰好 2 个 `*` 聚合位点，但得到 {len(star_atoms)} 个：{p_smiles}")

    star_idx0, star_idx1 = star_atoms[0].GetIdx(), star_atoms[1].GetIdx()

    # `*` 原子通常只连一个邻居：`* - X`
    nei0 = star_atoms[0].GetNeighbors()
    nei1 = star_atoms[1].GetNeighbors()
    if len(nei0) != 1 or len(nei1) != 1:
        raise ValueError(f"`*` 原子邻居数异常（每个 `*` 应该只有 1 个邻居）：{p_smiles}")

    n0, n1 = nei0[0].GetIdx(), nei1[0].GetIdx()

    # 闭环键型：尽量沿用 `*`-邻居 的键型（并要求两个端点一致），不一致则退化为单键
    bond_type = Chem.BondType.SINGLE
    if keep_bond_type:
        b0 = mol.GetBondBetweenAtoms(star_idx0, n0)
        b1 = mol.GetBondBetweenAtoms(star_idx1, n1)
        if b0 is not None and b1 is not None and b0.GetBondType() == b1.GetBondType():
            bond_type = b0.GetBondType()

    ed_mol = Chem.EditableMol(mol)

    # 如果端点已经有键，避免重复加键导致异常
    if mol.GetBondBetweenAtoms(n0, n1) is None:
        ed_mol.AddBond(int(n0), int(n1), order=bond_type)

    # 删除两个 `*`（注意索引会移动，必须从大到小删）
    for idx in sorted([star_idx0, star_idx1], reverse=True):
        ed_mol.RemoveAtom(int(idx))

    final_mol = ed_mol.GetMol()

    if sanitize:
        # SanitizeMol 可能抛异常（价态/芳香性等），交给调用方处理更清晰
        Chem.SanitizeMol(final_mol)

    smiles_out = Chem.MolToSmiles(final_mol)
    if return_mol:
        return smiles_out, final_mol
    return smiles_out


if __name__ == "__main__":
    # 极简自检样例：线性链首尾闭环 -> 环己烷
    print(cyclize_p_smiles("*CCCCC*"))

