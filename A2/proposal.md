<center style = "font-size: 3em"><b>Proposal: Portable High-Performence Microscope</b></center>

**姓名**：<u>毕邹彬</u>&emsp;**学号**：<u>3180105490</u>&emsp;**专业**：<u>计算机科学与技术</u>

**姓名**：<u>陈希尧</u>&emsp;**学号**：<u>3180103012</u>&emsp;**专业**：<u>计算机科学与技术</u>

**姓名**：<u>肖瑞轩</u>&emsp;**学号**：<u>3180103127</u>&emsp;**专业**：<u>计算机科学与技术</u>

**时间**：<u>2020-9-29</u>

**指导老师**：<u>吴鸿智</u>

<center style = "font-size: 2em">Table of Contents</center>

[TOC]

## Introduction

我们的计划是制作一款便携简易且高效的显微相机。我们打算在硬件和软件两方面进行实现。硬件方面，我们准备利用增加额外的透镜和机械调节装置来达到显微镜的放大倍数。软件方面，我们打算利用神经网络来弥补物理成像所产生的不足，例如色散、模糊（由抖动、边缘焦距不足等引起的模糊现象）等现象。

## Related Work

Simultaneous Acquisition of Microscale Reflectance and Normals[<sup>1</sup>](#refer-anchor-1)

Turn Your Smartphone Into a Digital Microscope! [<sup>2 </sup>](#refer-anchor-2)

## Our Approach

物理方面，我们打算先尝试使用淘宝上能够购买到的透镜来作为额外添加的镜头；光源可以使用LED光源通过Arduino来精确控制；搭载平台通过重构原本提供的相机支架实现平台的升降，增加一定的防抖动设备减少抖动产生的影响。

软件方面，由于直接获得的图像会有各种噪声、模糊的现象，我们准备利用深度学习来处理削弱这些影响，提高图片的清晰程度与真实程度，减少抖动、色散等产生的影响。

## Time line

- 9.29立项
- 10.9调研
- 10.16探究相机构造
- 10.23物理框架搭建，代码框架
- 10.30边写边调试
- 11.6最后测试
- 11.13完成报告与展示

## References

<div id="refer-anchor-1"></div>- [1] Giljoo Nam, Joo Ho Lee, Hongzhi Wu, Diego Gutierrez, Min H. Kim

Simultaneous Acquisition of Microscale Reflectance and Normals

SIGGRAPHAsia,2016

<div id="refer-anchor-2"></div>-[2] https://www.youtube.com/watch?v=KpMTkr_aiYU&feature=youtu.be