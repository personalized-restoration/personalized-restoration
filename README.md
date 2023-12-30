### Personalized Restoration via Dual-Pivot Tuning

Pradyumna Chari<sup>1</sup>, Sizhuo Ma<sup>2</sup>, Daniil Ostashev<sup>2</sup>, Achuta Kadambi<sup>1</sup>, Gurunandan Krishnan<sup>2</sup>, Jian Wang<sup>2</sup>\*, Kfir Aberman<sup>2</sup>

<sup>1</sup>University of California, Los Angeles, <sup>2</sup>Snap Inc., *Corresponding Author

![teaser](assets/teaser.gif)

Abstract: *Generative diffusion models can serve as a prior which ensures that solutions of image restoration systems adhere to the manifold of natural images. However, for restoring facial images, a personalized prior is necessary to accurately represent and reconstruct unique facial features of a given individual. In this paper, we propose a simple, yet effective, method for personalized restoration, called Dual-Pivot Tuning - a two-stage approach that personalize a blind restoration system while maintaining the integrity of the general prior and the distinct role of each component. Our key observation is that for optimal personalization, the generative model should be tuned around a fixed text pivot, while the guiding network should be tuned in a generic (non-personalized) manner, using the personalized generative model as a fixed "pivot". This approach ensures that personalization does not interfere with the restoration process, resulting in a natural appearance with high fidelity to the person's identity and the attributes of the degraded image. We evaluated our approach both qualitatively and quantitatively through extensive experiments with images of widely recognized individuals, comparing it against relevant baselines. Surprisingly, we found that our personalized prior not only achieves higher fidelity to identity with respect to the person's identity, but also outperforms state-of-the-art generic priors in terms of general image quality.*

## Code will be released in 14-30 days.


