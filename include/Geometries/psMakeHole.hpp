//   PSPtrType createTVMask() {
//     auto mask = LSPtrType::New(bounds, boundaryCons, gridDelta);
//     auto grid = mask->getGrid();

//     NumericType normal[3] = {0., 0., 1.};
//     NumericType origin[3] = {0., 0., maskHeight};

//     lsMakeGeometry<NumericType, 3>(
//         mask, lsSmartPointer<lsPlane<NumericType, 3>>::New(origin, normal))
//         .apply();
//     normal[2] = -1.0;
//     origin[2] = -gridDelta;
//     auto maskBottom = lsSmartPointer<lsDomain<NumericType, 3>>::New(grid);
//     lsMakeGeometry<NumericType, 3>(
//         maskBottom,
//         lsSmartPointer<lsPlane<NumericType, 3>>::New(origin, normal))
//         .apply();
//     lsBooleanOperation<NumericType, 3>(mask, maskBottom,
//                                        lsBooleanOperationEnum::INTERSECT)
//         .apply();

//     auto maskHole = lsSmartPointer<lsDomain<NumericType, 3>>::New(grid);

//     origin[2] = -2 * gridDelta;
//     normal[2] = 1.0;

//     lsMakeGeometry<NumericType, 3>(
//         maskHole, lsSmartPointer<lsCylinder<NumericType, 3>>::New(
//                       origin, normal, maskHeight + 3 * gridDelta, TVRadius))
//         .apply();

//     lsBooleanOperation<NumericType, 3>(
//         mask, maskHole, lsBooleanOperationEnum::RELATIVE_COMPLEMENT)
//         .apply();

//     auto substrate = makePlane(0.);
//     lsBooleanOperation<NumericType, 3>(substrate->getLevelSets()->back(),
//     mask,
//                                        lsBooleanOperationEnum::UNION)
//         .apply();

//     auto geometry = PSPtrType::New(mask);
//     geometry->insertNextLevelSet(substrate->getLevelSets()->back(), false);

//     return geometry;
//   }